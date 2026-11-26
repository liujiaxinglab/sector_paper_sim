"""Contract tests for the four daily steps and the opening orders they defer."""

from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import pandas as pd
from pyasn1.codec.der import decoder as der_decoder, encoder as der_encoder
from pyasn1.type import univ, useful
from pyasn1_modules import rfc3161, rfc5280, rfc5652
import pytest

from src.data_loader import (
    CHINA_TZ,
    MarketBundle,
    RegimeHistory,
    build_forward_adjusted_daily,
    compute_live_attestation_payload,
    compute_regime_decision_sha256,
    load_sim_config,
)
from src.engine import SimulationEngine, run_simulation
from src.strategies import (
    GridOrder,
    GridState,
    TurtleConfig,
    calc_commission,
    compute_grid_step,
    compute_symbol_indicators,
    evaluate_turtle_signal,
    match_grid_bar,
    max_affordable_shares,
)

START = "2026-11-23"
BENCHMARK = "159330.SZ"
SECTOR = "510720.SH"
OTHER = "159622.SZ"
WARMUP = pd.bdate_range(end="2026-11-20", periods=60).strftime("%Y-%m-%d").tolist()


class _TestTSA:
    """Test-only offline RFC 3161 TSA issuer."""

    def __init__(self, cn: str = "Engine Test TSA") -> None:
        base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.root_key = ec.generate_private_key(ec.SECP256R1())
        self.tsa_key = ec.generate_private_key(ec.SECP256R1())
        root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{cn} Root")])
        tsa_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{cn} Signer")])
        self.root_cert = (
            x509.CertificateBuilder()
            .subject_name(root_name).issuer_name(root_name)
            .public_key(self.root_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(base_time)
            .not_valid_after(base_time + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False, content_commitment=False, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=True,
                    crl_sign=True, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(self.root_key.public_key()), critical=False)
            .sign(self.root_key, hashes.SHA256())
        )
        self.tsa_cert = (
            x509.CertificateBuilder()
            .subject_name(tsa_name).issuer_name(root_name)
            .public_key(self.tsa_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(base_time)
            .not_valid_after(base_time + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, content_commitment=True, key_encipherment=False,
                    data_encipherment=False, key_agreement=False, key_cert_sign=False,
                    crl_sign=False, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(self.tsa_key.public_key()), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(
                    self.root_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
                ),
                critical=False,
            )
            .sign(self.root_key, hashes.SHA256())
        )

    def sign_digest(self, digest_bytes: bytes, gen_time: datetime) -> bytes:
        gen_utc = gen_time.astimezone(timezone.utc)
        tst = rfc3161.TSTInfo()
        tst["version"] = 1
        tst["policy"] = univ.ObjectIdentifier("1.2.3.4.5")
        tst["messageImprint"]["hashAlgorithm"]["algorithm"] = univ.ObjectIdentifier("2.16.840.1.101.3.4.2.1")
        tst["messageImprint"]["hashAlgorithm"]["parameters"] = univ.Null("")
        tst["messageImprint"]["hashedMessage"] = digest_bytes
        tst["serialNumber"] = 1
        tst["genTime"] = useful.GeneralizedTime(gen_utc.strftime("%Y%m%d%H%M%SZ"))
        e_content = der_encoder.encode(tst)

        sa_set = rfc5652.SignedAttributes()
        a1 = rfc5652.Attribute()
        a1["attrType"] = rfc5652.id_contentType
        a1["attrValues"][0] = der_encoder.encode(univ.ObjectIdentifier("1.2.840.113549.1.9.16.1.4"))
        a2 = rfc5652.Attribute()
        a2["attrType"] = rfc5652.id_messageDigest
        a2["attrValues"][0] = der_encoder.encode(univ.OctetString(hashlib.sha256(e_content).digest()))
        sa_set[0] = a1
        sa_set[1] = a2
        sig = self.tsa_key.sign(der_encoder.encode(sa_set), ec.ECDSA(hashes.SHA256()))

        sd = rfc5652.SignedData()
        sd["version"] = 3
        da = rfc5652.DigestAlgorithmIdentifier()
        da["algorithm"] = univ.ObjectIdentifier("2.16.840.1.101.3.4.2.1")
        da["parameters"] = univ.Null("")
        sd["digestAlgorithms"][0] = da
        sd["encapContentInfo"]["eContentType"] = univ.ObjectIdentifier("1.2.840.113549.1.9.16.1.4")
        sd["encapContentInfo"]["eContent"] = e_content
        cert_choice = rfc5652.CertificateChoices()
        parsed_cert, _ = der_decoder.decode(
            self.tsa_cert.public_bytes(serialization.Encoding.DER), asn1Spec=rfc5280.Certificate()
        )
        cert_choice["certificate"] = parsed_cert
        sd["certificates"][0] = cert_choice

        si = rfc5652.SignerInfo()
        si["version"] = 1
        issuer_asn1, _ = der_decoder.decode(self.tsa_cert.issuer.public_bytes(), asn1Spec=rfc5280.Name())
        si["sid"]["issuerAndSerialNumber"]["issuer"] = issuer_asn1
        si["sid"]["issuerAndSerialNumber"]["serialNumber"] = self.tsa_cert.serial_number
        si["digestAlgorithm"] = da
        si["signedAttrs"][0] = a1
        si["signedAttrs"][1] = a2
        si["signatureAlgorithm"]["algorithm"] = univ.ObjectIdentifier("1.2.840.10045.4.3.2")
        si["signature"] = sig
        sd["signerInfos"][0] = si

        ci = rfc5652.ContentInfo()
        ci["contentType"] = rfc5652.id_signedData
        ci["content"] = der_encoder.encode(sd)
        resp = rfc3161.TimeStampResp()
        resp["status"]["status"] = 0
        resp["timeStampToken"] = ci
        return der_encoder.encode(resp)


@pytest.fixture()
def trusted_test_tsa(monkeypatch: pytest.MonkeyPatch) -> _TestTSA:
    tsa = _TestTSA()
    monkeypatch.setattr("src.data_loader._load_default_trusted_tsa_roots", lambda: (tsa.root_cert,))
    return tsa


def _attach_in_memory_live_proof(symbol: str, rec: dict[str, Any], tsa: _TestTSA) -> dict[str, Any]:
    recorded_cst = datetime.fromisoformat(rec["recorded_at"].replace("Z", "+00:00")).astimezone(CHINA_TZ).isoformat()
    decision_sha = compute_regime_decision_sha256({"symbol": symbol, **rec}, recorded_cst)
    attestation_bytes = compute_live_attestation_payload(
        record_id=rec["record_id"],
        decision_date=rec["decision_date"],
        effective_date=rec["effective_date"],
        symbol=symbol,
        prev_regime=rec["prev_regime"],
        new_regime=rec["regime"],
        recorded_at_cst=recorded_cst,
        decision_sha256=decision_sha,
        decision_entry_sha256=None,
        decision_ref=rec.get("decision_ref"),
        card_path=rec.get("card_path"),
    )
    tsr_bytes = tsa.sign_digest(hashlib.sha256(attestation_bytes).digest(), datetime.fromisoformat(recorded_cst))
    return {
        **rec,
        "pre_market_proof": {
            "decision_sha256": decision_sha,
            "tsa_token_b64": base64.b64encode(tsr_bytes).decode("ascii"),
        },
    }


def sessions(count: int) -> list[str]:
    return pd.bdate_range(start=START, periods=count).strftime("%Y-%m-%d").tolist()


def make_bundle(
    days: list[str],
    *,
    sectors: tuple[str, ...] = (SECTOR,),
    bars: dict[str, dict[str, tuple[float, float, float, float] | None]] | None = None,
    dividends: dict[str, dict[str, float]] | None = None,
    splits: dict[str, list[dict[str, Any]]] | None = None,
    initial_routes: dict[str, str] | None = None,
    route_changes: dict[str, list[tuple[str, str]]] | None = None,
    record_overrides: dict[str, list[dict[str, Any]]] | None = None,
    forward_eval_start_date: str | None = None,
    backfill_replay_window: dict[str, str] | None | object = ...,
    tsa: _TestTSA | None = None,
) -> MarketBundle:
    config = load_sim_config()
    config["forward_eval_start_date"] = forward_eval_start_date
    if backfill_replay_window is not ...:
        config["backfill_replay_window"] = backfill_replay_window
    symbols = [BENCHMARK, *sectors]
    config["universe"] = {"benchmark": BENCHMARK, "sectors": list(sectors)}
    all_days = WARMUP + days
    raw = {}
    for symbol in symbols:
        overrides = (bars or {}).get(symbol, {})
        rows = []
        for day in all_days:
            ohlc = overrides.get(day, (1.0, 1.005, 0.995, 1.0))
            if ohlc is not None:
                rows.append((day, *ohlc, 10000, 10000))
        raw[symbol] = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "amount", "volume"])
    split_map = {symbol: (splits or {}).get(symbol, []) for symbol in symbols}
    dividend_map = {symbol: (dividends or {}).get(symbol, {}) for symbol in symbols}
    routes = {symbol: (initial_routes or {}).get(symbol, "TRANSITION") for symbol in sectors}
    if record_overrides is not None:
        records_by_symbol = {
            sym: [
                _attach_in_memory_live_proof(sym, rec, tsa)
                if tsa is not None and rec.get("record_type") == "LIVE" and "pre_market_proof" not in rec
                else rec
                for rec in recs
            ]
            for sym, recs in record_overrides.items()
        }
    else:
        records_by_symbol = {}
        for symbol, changes in (route_changes or {}).items():
            prev_reg = routes[symbol]
            sym_records = []
            for idx, (day, regime) in enumerate(changes, 1):
                day_idx = all_days.index(day)
                dec_day = all_days[day_idx - 1]
                rec = {
                    "record_id": f"RH-{symbol}-{idx}",
                    "decision_date": dec_day,
                    "effective_date": day,
                    "prev_regime": prev_reg,
                    "regime": regime,
                    "record_type": "LIVE",
                    "recorded_at": f"{dec_day}T16:00:00+08:00",
                }
                if tsa is not None:
                    rec = _attach_in_memory_live_proof(symbol, rec, tsa)
                sym_records.append(rec)
                prev_reg = regime
            records_by_symbol[symbol] = sym_records
    history = RegimeHistory("2.0.0", START, routes, records_by_symbol)
    return MarketBundle(
        config=config, benchmark=BENCHMARK, sectors=list(sectors), symbols=symbols, bars=raw,
        adjusted_bars={
            symbol: build_forward_adjusted_daily(
                raw[symbol], split_map[symbol], dividend_map[symbol], cutoff_date=days[-1]
            )
            for symbol in symbols
        },
        watchlist={symbol: {"name": symbol, "current_regime": "TRAP_OFF"} for symbol in symbols},
        dividends=dividend_map, splits=split_map, regime_history=history,
        common_end_date=days[-1], common_trading_dates=WARMUP + days,
        authoritative_trading_dates=WARMUP + days, suspension_dates_by_symbol={}, lagged_dates_by_symbol={},
        metadata={"corporate_actions_as_of": days[-1]},
    )


def trades(day: dict[str, Any], account: str, symbol: str | None = None) -> list[dict[str, Any]]:
    return [
        trade for trade in day["trades"]
        if trade["account"] == account and (symbol is None or trade["symbol"] == symbol)
    ]


def resume(engine: SimulationEngine) -> SimulationEngine:
    state = json.loads(json.dumps(engine.export_state(), allow_nan=False))
    return SimulationEngine.from_state(engine.bundle, state)


@pytest.mark.parametrize(
    "budget,price,expected",
    [(110.0, 1.1, 0), (110.09, 1.1, 0), (110.1, 1.1, 100), (10000.0, 1.0, 9900), (10002.0, 1.0, 10000)],
)
def test_lot_sizing_includes_minimum_and_proportional_commission(budget, price, expected):
    assert max_affordable_shares(budget, price) == expected
    assert calc_commission(499.0) == 0.10
    assert calc_commission(10000.0) == 2.0


@pytest.mark.parametrize("per_unit,open_price,quantity", [(0.001, 0.999, 0), (0.0011, 1.0, 100), (0.025, 1.0, 2400)])
def test_ex_date_dividend_reinvests_only_within_payout_after_fees(per_unit, open_price, quantity):
    first, ex_day = sessions(2)
    bundle = make_bundle(
        [first, ex_day],
        bars={BENCHMARK: {ex_day: (open_price, 1.02, 0.98, 1.01)}},
        dividends={BENCHMARK: {ex_day: per_unit}},
    )
    engine = SimulationEngine(bundle)
    initial = engine.step_day(first)["accounts"]["D"]
    day = engine.step_day(ex_day)
    amount = 99900 * per_unit
    assert [event["event_type"] for event in day["events"]] == ["DIVIDEND_CASH"]
    assert day["accounts"]["D"]["dividend_cash"] == pytest.approx(amount)
    reinvest = trades(day, "D")
    if quantity:
        assert len(reinvest) == 1
        assert reinvest[0]["action"] == "DIVIDEND_REINVEST"
        assert reinvest[0]["shares"] == quantity
        assert reinvest[0]["price"] == open_price
        assert reinvest[0]["commission"] == calc_commission(quantity * open_price)
    else:
        assert reinvest == []
        assert initial["cash"] + amount > 100 * open_price + 0.10
    cash, market_value, equity = (day["accounts"]["D"][key] for key in ("cash", "market_value", "equity"))
    assert cash - initial["cash"] == pytest.approx(amount + sum(t["net_cash_flow"] for t in reinvest))
    assert market_value == pytest.approx((99900 + quantity) * 1.01)
    assert equity == pytest.approx(cash + market_value)
    assert equity - initial["equity"] == pytest.approx(market_value - initial["market_value"] + cash - initial["cash"])


@pytest.mark.parametrize("target,action", [("TRANSITION", "REGIME_SWITCH_EXIT"), ("TRAP_OFF", "TRAP_OFF_LIQUIDATION")])
def test_route_to_cash_liquidates_at_open_and_keeps_dividend(target, action, trusted_test_tsa):
    first, switch = sessions(2)
    bundle = make_bundle(
        [first, switch], initial_routes={SECTOR: "RANGE_GRID"},
        bars={SECTOR: {switch: (0.8, 1.05, 0.7, 0.95)}},
        route_changes={SECTOR: [(switch, target)]},
        dividends={SECTOR: {switch: 0.03}},
        tsa=trusted_test_tsa,
    )
    engine = SimulationEngine(bundle)
    engine.step_day(first)
    account = engine.accounts["A"]
    held = account.get_shares(SECTOR)
    day = engine.step_day(switch)
    assert [(t["action"], t["price"], t["shares"]) for t in trades(day, "A")] == [(action, 0.8, held)]
    assert day["accounts"]["A"]["dividend_cash"] == pytest.approx(held * 0.03)
    assert account.get_shares(SECTOR) == 0
    state = account.symbol_states[SECTOR]
    assert state.grid is None and state.active_regime == target


def test_cross_strategy_switch_exits_before_initializing_at_same_open(trusted_test_tsa):
    first, grid_day, trend_day = sessions(3)
    bundle = make_bundle(
        [first, grid_day, trend_day], initial_routes={SECTOR: "TREND_UP"},
        route_changes={SECTOR: [(grid_day, "RANGE_GRID"), (trend_day, "TREND_UP")]},
        bars={SECTOR: {grid_day: (1.15, 1.16, 1.14, 1.155), trend_day: (0.9, 0.91, 0.89, 0.9)}},
        tsa=trusted_test_tsa,
    )
    engine = run_simulation(bundle)
    assert [t["action"] for t in trades(engine.daily_results[0], "A")] == ["TURTLE_ENTRY_BUY"]
    assert [(t["action"], t["price"]) for t in trades(engine.daily_results[1], "A")] == [
        ("REGIME_SWITCH_EXIT", 1.15), ("GRID_INIT_BUY", 1.15),
    ]
    assert [(t["action"], t["price"]) for t in trades(engine.daily_results[2], "A")] == [
        ("REGIME_SWITCH_EXIT", 0.9), ("TURTLE_ENTRY_BUY", 0.9),
    ]
    assert engine.accounts["A"].symbol_states[SECTOR].grid is None


def test_turtle_sizes_by_budget_and_trades_the_open_after_the_signal():
    first, next_day, exit_day = sessions(3)
    sector_bars = {day: (1.0, 1.6, 0.5, 1.0) for day in WARMUP}
    sector_bars.update({
        WARMUP[-1]: (1.0, 1.6, 0.5, 0.99),
        first: (0.96, 1.25, 0.95, 1.20),
        next_day: (0.85, 1.26, 0.81, 0.90),
        exit_day: (1.05, 1.06, 1.04, 1.055),
    })
    engine = run_simulation(make_bundle(
        [first, next_day, exit_day],
        bars={SECTOR: sector_bars}, initial_routes={SECTOR: "TREND_UP"},
    ))
    for account in ("A", "B1"):
        assert trades(engine.daily_results[0], account) == []
        buy, = trades(engine.daily_results[1], account)
        sell, = trades(engine.daily_results[2], account)
        assert (buy["action"], buy["price"], buy["shares"], buy["trigger_price"]) == (
            "TURTLE_ENTRY_BUY", 0.85, 19600, 1.0,
        )
        assert (sell["action"], sell["price"], sell["shares"], sell["trigger_price"]) == (
            "TURTLE_EXIT_SELL", 1.05, 19600, 0.99,
        )


def test_turtle_channel_excludes_signal_bar_and_uses_close_inclusive_limits():
    frame = pd.DataFrame({"close": [7.0] + [1.0] * 55 + [1.0], "high": [8.0] * 57, "low": [0.5] * 57})
    turtle = TurtleConfig()
    assert evaluate_turtle_signal(compute_symbol_indicators(frame.iloc[-55:], turtle, 20), 0, 0) is None
    indicators = compute_symbol_indicators(frame, turtle, 20)
    assert evaluate_turtle_signal(indicators, 0, 0).side == "BUY"
    assert evaluate_turtle_signal(indicators, 100, 100).side == "SELL"
    assert evaluate_turtle_signal(indicators, 100, 0) is None


def test_turtle_cash_limit_uses_sorted_symbol_order():
    first, second = sessions(2)
    sectors = ("515790.SH", "512880.SH", "512480.SH", "512400.SH", SECTOR, OTHER)
    last = sectors[0]
    engine = run_simulation(make_bundle(
        [first, second], sectors=sectors,
        bars={
            **{symbol: {first: (1.0, 2.0, 1.0, 2.0)} for symbol in sectors if symbol != last},
            last: {WARMUP[-1]: (1.0, 1.0, 0.9, 0.99), first: (1.0, 1.1, 1.0, 1.1)},
        },
    ))
    day1, day2 = engine.daily_results
    assert [t["symbol"] for t in trades(day1, "B1")] == sorted(set(sectors) - {last})
    assert day1["accounts"]["B1"]["equity"] / 6 > day1["accounts"]["B1"]["cash"]
    buy, = trades(day2, "B1")
    assert buy["symbol"] == last
    assert buy["shares"] == 16900


def test_today_dividend_shares_are_locked_even_when_turtle_exits():
    first, ex_day = sessions(2)
    engine = run_simulation(make_bundle(
        [first, ex_day], dividends={SECTOR: {ex_day: 0.02}},
        bars={SECTOR: {first: (1.0, 1.0, 0.9, 0.9)}},
    ))
    first_buy, = trades(engine.daily_results[0], "B1")
    buy, sell = trades(engine.daily_results[1], "B1")
    assert buy["action"] == "DIVIDEND_REINVEST"
    assert sell["action"] == "TURTLE_EXIT_SELL"
    assert buy["shares"] == 300 and sell["shares"] == first_buy["shares"]
    assert engine.accounts["B1"].get_shares(SECTOR) == 300


def test_grid_initializes_at_open_with_half_budget_and_symmetric_arithmetic_levels():
    bundle = make_bundle(
        [START], bars={SECTOR: {START: (1.1, 1.11, 1.09, 1.105)}},
        initial_routes={SECTOR: "RANGE_GRID"},
    )
    engine = run_simulation(bundle)
    for account_id in ("A", "B2"):
        account = engine.accounts[account_id]
        grid = account.symbol_states[SECTOR].grid
        assert grid.anchor_price == 1.1
        assert grid.step == 0.015
        assert [order.level for order in grid.pending_orders] == [-4, -3, -2, -1, 1, 2, 3, 4]
        assert [order.price for order in grid.pending_orders] == pytest.approx(
            [1.034, 1.0505, 1.067, 1.0835, 1.1165, 1.133, 1.1495, 1.166]
        )
        buy, = trades(engine.daily_results[0], account_id)
        assert (buy["action"], buy["price"], buy["shares"]) == ("GRID_INIT_BUY", 1.1, 7500)
        assert sum(order.shares for order in grid.pending_orders) == 7500


def test_grid_atr_excludes_today_and_uses_adjusted_prior_close_without_cap():
    split_day = WARMUP[-5]
    sector_bars = {day: (2.0, 2.1, 1.9, 2.0) for day in WARMUP}
    sector_bars.update({day: (1.0, 1.05, 0.95, 1.0) for day in WARMUP[-5:]})
    sector_bars[START] = (1.5, 1.85, 1.49, 1.8)
    engine = run_simulation(make_bundle(
        [START], bars={SECTOR: sector_bars},
        splits={SECTOR: [{"effective_date": split_day, "factor": 0.5}]},
    ))
    grid = engine.accounts["B2"].symbol_states[SECTOR].grid
    assert grid.anchor_price == 1.5
    assert grid.step == pytest.approx(0.1)
    assert compute_grid_step(atr20=0.0, previous_close=1.0) == 0.015
    assert compute_grid_step(atr20=0.3, previous_close=1.0) == 0.3


def grid_book() -> GridState:
    return GridState(1.0, 0.05, 110.0, START, (
        GridOrder("BUY", -2, 0.9),
        GridOrder("BUY", -1, 0.95),
        GridOrder("SELL", 1, 1.05, 100),
        GridOrder("SELL", 2, 1.1, 100),
    ))


@pytest.mark.parametrize(
    "close,expected", [
        (1.04, [("BUY", -1), ("BUY", -2), ("SELL", 1), ("SELL", 2)]),
        (1.00, [("BUY", -1), ("BUY", -2), ("SELL", 1), ("SELL", 2)]),
        (0.96, [("SELL", 1), ("SELL", 2), ("BUY", -1), ("BUY", -2)]),
    ],
)
def test_grid_candle_path_matches_both_sides_and_defers_counter_orders(close, expected):
    grid, fills = match_grid_bar(
        grid_state=grid_book(), open_price=1.0, high_price=1.11, low_price=0.89,
        close_price=close, sellable_shares=200, available_cash=300,
    )
    assert [(fill.side, fill.level) for fill in fills] == expected
    assert [(order.level, order.side) for order in grid.pending_orders] == [
        (-1, "SELL"), (0, "BUY"), (0, "SELL"), (1, "BUY"),
    ]
    _, next_fills = match_grid_bar(
        grid_state=grid, open_price=1.0, high_price=1.0, low_price=1.0,
        close_price=1.0, sellable_shares=200, available_cash=300,
    )
    assert [(fill.side, fill.level) for fill in next_fills] == [("BUY", 1), ("BUY", 0), ("SELL", -1), ("SELL", 0)]


@pytest.mark.parametrize("close,expected", [(1.02, ["SELL"]), (0.97, ["SELL", "BUY"])])
def test_grid_first_leg_cash_matters_on_a_wide_candle(close, expected):
    _, fills = match_grid_bar(
        grid_state=grid_book(), open_price=1.0, high_price=1.06, low_price=0.94,
        close_price=close, sellable_shares=100, available_cash=0,
    )
    assert [fill.side for fill in fills] == expected


@pytest.mark.parametrize("open_price,side", [(0.86, "BUY"), (1.2, "SELL")])
def test_grid_gap_price_improves_existing_limits(open_price, side):
    _, fills = match_grid_bar(
        grid_state=grid_book(), open_price=open_price, high_price=open_price + 0.01,
        low_price=open_price - 0.01, close_price=open_price, sellable_shares=200, available_cash=300,
    )
    assert [(fill.side, fill.exec_price) for fill in fills] == [(side, open_price), (side, open_price)]


def test_grid_buy_leg_stops_at_first_unaffordable_level():
    grid = grid_book()
    state, fills = match_grid_bar(
        grid_state=grid, open_price=0.99, high_price=1.0, low_price=0.88,
        close_price=0.99, sellable_shares=200, available_cash=91,
    )
    assert fills == []
    assert state == grid


def test_grid_cannot_sell_opening_or_intraday_buys_until_next_market_day():
    first, second = sessions(2)
    engine = run_simulation(make_bundle(
        [first, second], bars={SECTOR: {
            first: (1.0, 1.02, 0.98, 1.005),
            second: (1.01, 1.02, 1.0, 1.005),
        }},
    ))
    assert [t["action"] for t in trades(engine.daily_results[0], "B2")] == ["GRID_INIT_BUY", "GRID_BUY"]
    sells = trades(engine.daily_results[1], "B2")
    assert [(t["action"], t["grid_level"]) for t in sells] == [("GRID_SELL", 0), ("GRID_SELL", 1)]


def test_nonpositive_grid_buy_levels_are_skipped_without_a_step_cap():
    bars = {day: (1.0, 1.3, 0.7, 1.0) for day in WARMUP}
    engine = run_simulation(make_bundle([START], bars={SECTOR: bars}))
    grid = engine.accounts["B2"].symbol_states[SECTOR].grid
    assert grid.step == pytest.approx(0.6)
    assert [order.level for order in grid.pending_orders] == [-1, 1, 2, 3, 4]


def test_suspension_applies_split_and_payout_then_resumes_liquidation_and_reinvestment(trusted_test_tsa):
    first, halted, still_halted, reopened = sessions(4)
    bundle = make_bundle(
        [first, halted, still_halted, reopened], initial_routes={SECTOR: "RANGE_GRID"},
        route_changes={SECTOR: [(halted, "TRANSITION")]},
        bars={SECTOR: {halted: None, still_halted: None, reopened: (1.5, 1.51, 1.49, 1.5)}},
        splits={SECTOR: [{"effective_date": halted, "factor": 1.5}]},
        dividends={SECTOR: {halted: 0.1, still_halted: 0.05}},
        tsa=trusted_test_tsa,
    )
    engine = SimulationEngine(bundle)
    engine.step_day(first)
    previous_position = engine.accounts["A"].positions[SECTOR].avg_cost
    previous_grid = engine.accounts["B2"].symbol_states[SECTOR].grid
    day = engine.step_day(halted)
    assert day["suspended"] == [SECTOR]
    assert day["closes"][SECTOR] == 1.0
    assert trades(day, "A") == trades(day, "B2") == []

    held = 5533
    for account_id in ("A", "B2"):
        account = engine.accounts[account_id]
        assert account.get_shares(SECTOR) == held
        assert account.positions[SECTOR].avg_cost == pytest.approx(previous_position * 1.5)
        assert day["accounts"][account_id]["dividend_cash"] == pytest.approx(held * 0.1)
        assert day["accounts"][account_id]["market_value"] == held * 1.0
    assert engine.accounts["A"].symbol_states[SECTOR].active_regime == "RANGE_GRID"
    grid = engine.accounts["B2"].symbol_states[SECTOR].grid
    assert grid.anchor_price == 1.5
    for old, new in zip(previous_grid.pending_orders, grid.pending_orders, strict=True):
        assert new.price == pytest.approx(old.price * 1.5)
        assert new.shares == round(old.shares / 1.5)

    engine.step_day(still_halted)
    saved = engine.export_state()["accounts"]
    assert saved["A"]["symbol_states"][SECTOR]["pending_regime"] == "TRANSITION"
    assert saved["A"]["symbol_states"][SECTOR]["pending_liquidation"] == "REGIME_SWITCH_EXIT"
    assert saved["B2"]["symbol_states"][SECTOR]["pending_dividend"] == pytest.approx(held * 0.15)
    engine = resume(engine)
    day = engine.step_day(reopened)
    liquidation, = trades(day, "A")
    assert (liquidation["action"], liquidation["shares"], liquidation["price"]) == ("REGIME_SWITCH_EXIT", held, 1.5)
    reinvest, = trades(day, "B2")
    assert (reinvest["action"], reinvest["shares"], reinvest["price"]) == ("DIVIDEND_REINVEST", 500, 1.5)
    assert reinvest["commission"] == pytest.approx(0.15)
    assert engine.accounts["B2"].symbol_states[SECTOR].pending_dividend == 0


def test_pending_route_exit_survives_another_target_change_before_reopening(trusted_test_tsa):
    first, halted, reopened = sessions(3)
    engine = SimulationEngine(make_bundle(
        [first, halted, reopened], initial_routes={SECTOR: "RANGE_GRID"},
        route_changes={SECTOR: [(halted, "TRAP_OFF"), (reopened, "RANGE_GRID")]},
        bars={SECTOR: {halted: None, reopened: (1.2, 1.21, 1.19, 1.2)}},
        tsa=trusted_test_tsa,
    ))
    engine.step_day(first)
    engine.step_day(halted)
    engine = resume(engine)
    assert [(t["action"], t["price"]) for t in trades(engine.step_day(reopened), "A")] == [
        ("TRAP_OFF_LIQUIDATION", 1.2), ("GRID_INIT_BUY", 1.2),
    ]


def test_suspended_start_defers_grid_turtle_and_bh_per_symbol():
    first, reopened = sessions(2)
    engine = SimulationEngine(make_bundle(
        [first, reopened], sectors=(SECTOR, OTHER),
        initial_routes={SECTOR: "RANGE_GRID"},
        bars={
            SECTOR: {first: None, reopened: (1.3, 1.31, 1.29, 1.3)},
            BENCHMARK: {first: None, reopened: (2.0, 2.01, 1.99, 2.0)},
            OTHER: {first: (1.0, 1.5, 1.0, 1.5)},
        },
    ))
    day = engine.step_day(first)
    assert trades(day, "D") == []
    assert [t["symbol"] for t in trades(day, "C")] == [OTHER]
    assert engine.accounts["B1"].symbol_states[SECTOR].pending_turtle.side == "BUY"
    engine = resume(engine)
    resumed = engine.step_day(reopened)
    for account, action, price in [
        ("A", "GRID_INIT_BUY", 1.3), ("B1", "TURTLE_ENTRY_BUY", 1.3), ("B2", "GRID_INIT_BUY", 1.3),
        ("C", "BH_INIT_BUY", 1.3), ("D", "BH_INIT_BUY", 2.0),
    ]:
        symbol = BENCHMARK if account == "D" else SECTOR
        entry, = trades(resumed, account, symbol)
        assert (entry["action"], entry["price"]) == (action, price)
    c_trade, = trades(resumed, "C")
    assert c_trade["shares"] == max_affordable_shares(day["accounts"]["C"]["equity"] / 6, 1.3)
    assert trades(resumed, "D")[0]["shares"] == 49900


def test_bh_waits_until_an_initial_lot_can_be_bought_and_then_holds():
    first, affordable, later = sessions(3)
    engine = run_simulation(make_bundle(
        [first, affordable, later],
        bars={
            SECTOR: {first: (200.0, 200.0, 200.0, 200.0)},
            BENCHMARK: {first: (1000.0, 1000.0, 1000.0, 1000.0)},
        },
    ))
    for account in ("C", "D"):
        assert trades(engine.daily_results[0], account) == []
        fill, = trades(engine.daily_results[1], account)
        assert fill["action"] == "BH_INIT_BUY" and fill["price"] == 1.0
        assert trades(engine.daily_results[2], account) == []


def test_turtle_exit_is_queued_over_suspension_and_executed_at_reopened_open():
    first, halted, reopened = sessions(3)
    bundle = make_bundle(
        [first, halted, reopened],
        bars={SECTOR: {first: (1.0, 1.0, 0.7, 0.7), halted: None, reopened: (0.95, 0.97, 0.93, 0.94)}},
    )
    engine = SimulationEngine(bundle)
    buy, = trades(engine.step_day(first), "B1")
    day = engine.step_day(halted)
    assert not trades(day, "B1")
    assert engine.accounts["B1"].symbol_states[SECTOR].pending_turtle.side == "SELL"
    engine = resume(engine)
    sell, = trades(engine.step_day(reopened), "B1")
    assert (sell["action"], sell["price"], sell["shares"]) == ("TURTLE_EXIT_SELL", 0.95, buy["shares"])


def test_future_actions_and_bars_do_not_change_the_anchor_day():
    future = sessions(4)
    early = run_simulation(make_bundle([START], initial_routes={SECTOR: "RANGE_GRID"}))
    later = run_simulation(make_bundle(
        future, initial_routes={SECTOR: "RANGE_GRID"},
        splits={SECTOR: [{"effective_date": future[2], "factor": 0.5}]},
        dividends={SECTOR: {future[3]: 0.113}},
        bars={SECTOR: {future[2]: (0.5, 0.7, 0.4, 0.51), future[3]: (0.4, 0.6, 0.3, 0.45)}},
    ), end_date=START)
    assert later.daily_results == early.daily_results
    assert later.export_state() == early.export_state()


def test_equity_and_cash_conservation_checks_detect_invalid_changes(monkeypatch):
    bundle = make_bundle(sessions(2))
    engine = SimulationEngine(bundle)
    engine.step_day(START)
    engine.accounts["D"].equity += 1
    with pytest.raises(RuntimeError, match="Equity conservation"):
        engine.step_day(sessions(2)[-1])

    engine = SimulationEngine(bundle)
    original = engine._trade

    def inject_cash(day, *args, **kwargs):
        original(day, *args, **kwargs)
        day.account.cash += 0.01

    monkeypatch.setattr(engine, "_trade", inject_cash)
    with pytest.raises(RuntimeError, match="Cash conservation"):
        engine.step_day(START)


def test_engine_rejects_skipping_calendar_days_and_warmup_orders():
    second = sessions(2)[-1]
    engine = SimulationEngine(make_bundle([START, second]))
    for day in (second, (date.fromisoformat(START) - timedelta(days=3)).isoformat()):
        with pytest.raises(ValueError, match="next BS-01"):
            engine.step_day(day)
    assert engine.daily_results == []
    assert engine.last_processed_date is None


def test_engine_blocks_effective_unverified_event_on_direct_step_and_run():
    days = sessions(2)
    bundle = make_bundle(days)
    bundle.metadata["unverified_corporate_action_events"] = [{
        "date": days[1], "symbol": SECTOR, "kind": "split",
        "event_status": "PROVISIONAL", "evidence_gap": "missing official PDF",
    }]
    engine = SimulationEngine(bundle)
    engine.step_day(days[0])
    with pytest.raises(ValueError, match="1 个未 VERIFIED 公司行为"):
        engine.step_day(days[1])

    fresh = SimulationEngine(bundle)
    with pytest.raises(ValueError, match="1 个未 VERIFIED 公司行为"):
        fresh.run()
    assert fresh.daily_results == []


def test_invalid_ohlc_range_cannot_be_used_as_a_tradable_bar():
    bundle = make_bundle([START], bars={SECTOR: {START: (1.0, 1.01, 0.99, 1.02)}})
    with pytest.raises(ValueError, match="Invalid OHLC range"):
        SimulationEngine(bundle)


def test_formal_run_rejects_backfill_while_replay_allows_and_excludes_from_live_stats():
    first, second = sessions(2)
    bundle = make_bundle(
        [first, second],
        initial_routes={SECTOR: "TRANSITION"},
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-001",
                    "decision_date": first,
                    "effective_date": second,
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "4t03 backfill",
                }
            ]
        },
    )
    formal = SimulationEngine(bundle, run_mode="formal")
    with pytest.raises(ValueError, match="正式前瞻运行拒绝消费 BACKFILL"):
        formal.run()
    assert formal.daily_results == []

    formal.step_day(first)
    with pytest.raises(ValueError, match="正式前瞻运行拒绝消费 BACKFILL"):
        formal.step_day(second)

    replay = run_simulation(bundle, run_mode="replay")
    assert len(replay.daily_results) == 2
    assert replay.daily_results[0]["routed_positions"][SECTOR]["target_regime"] == "TRANSITION"
    assert replay.daily_results[1]["routed_positions"][SECTOR]["target_regime"] == "RANGE_GRID"
    stats = replay.daily_results[1]["routing_stats"]
    assert stats["live_records_applied"] == 0
    assert stats["backfill_records_excluded"] == 1
    assert stats["backfill_record_ids"] == ["RH-BF-001"]


def test_backfill_init_then_formal_live_resume_and_late_live_rejection(trusted_test_tsa):
    day1, day2, day3 = sessions(3)
    records = {
        SECTOR: [
            {
                "record_id": "RH-BF-001",
                "decision_date": day1,
                "effective_date": day2,
                "prev_regime": "TRANSITION",
                "regime": "TRANSITION",
                "record_type": "BACKFILL",
                "recorded_at": None,
                "time_evidence_boundary": "init backfill",
            },
            {
                "record_id": "RH-LIVE-001",
                "decision_date": day2,
                "effective_date": day3,
                "prev_regime": "TRANSITION",
                "regime": "RANGE_GRID",
                "record_type": "LIVE",
                "recorded_at": f"{day2}T16:00:00+08:00",
            },
        ]
    }
    init_bundle = make_bundle(
        [day1, day2, day3],
        initial_routes={SECTOR: "TRANSITION"},
        record_overrides=records,
        forward_eval_start_date=day3,
        tsa=trusted_test_tsa,
    )
    init_engine = run_simulation(init_bundle, end_date=day2, run_mode="replay")
    saved_state = json.loads(json.dumps(init_engine.export_state(), allow_nan=False))

    formal_engine = SimulationEngine.from_state(
        init_bundle, saved_state, run_mode="formal", forward_eval_start_date=day3
    )
    summary = formal_engine.run()
    assert summary["processed_dates"] == [day3]
    assert formal_engine.daily_results[0]["routed_positions"][SECTOR]["active_regime"] == "RANGE_GRID"
    stats = summary["routing_stats"]
    assert stats["live_records_applied"] == 1
    assert stats["forward_live_records_applied"] == 1
    assert stats["backfill_records_excluded"] == 1

    # Late LIVE timestamp (>= 09:30+08:00 on effective_date) must fail formal run
    late_records = json.loads(json.dumps(records))
    late_records[SECTOR][1]["recorded_at"] = f"{day3}T09:30:00+08:00"
    late_bundle = make_bundle(
        [day1, day2, day3],
        initial_routes={SECTOR: "TRANSITION"},
        record_overrides=late_records,
        forward_eval_start_date=day3,
        tsa=trusted_test_tsa,
    )
    with pytest.raises(ValueError, match="market open"):
        SimulationEngine.from_state(late_bundle, saved_state, run_mode="formal").run()


def test_direct_engine_formal_rejects_unproven_or_untrusted_live_records():
    """Regression (Blocking 1): direct run_simulation(..., run_mode='formal') with source_path=None must reject unproven/untrusted LIVE."""
    day1, day2 = sessions(2)
    unproven_bundle = make_bundle(
        [day1, day2],
        initial_routes={SECTOR: "TRANSITION"},
        route_changes={SECTOR: [(day2, "RANGE_GRID")]},
        tsa=None,
    )
    assert unproven_bundle.regime_history.source_path is None
    with pytest.raises(ValueError, match="pre_market_proof"):
        run_simulation(unproven_bundle, run_mode="formal")

    # Even with a self-signed TSA token + source_path=None, production trust roots reject ("unverified source")
    fake_tsa = _TestTSA("Untrusted Direct Engine TSA")
    untrusted_bundle = make_bundle(
        [day1, day2],
        initial_routes={SECTOR: "TRANSITION"},
        route_changes={SECTOR: [(day2, "RANGE_GRID")]},
        tsa=fake_tsa,
    )
    with pytest.raises(ValueError, match="unverified source"):
        run_simulation(untrusted_bundle, run_mode="formal")

    # Even with receipt_path + decision_sha256 without tsa_token_b64, direct engine rejects
    partial_bundle = make_bundle(
        [day1, day2],
        initial_routes={SECTOR: "TRANSITION"},
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-DIRECT-PARTIAL",
                    "decision_date": day1,
                    "effective_date": day2,
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "LIVE",
                    "recorded_at": f"{day1}T16:00:00+08:00",
                    "pre_market_proof": {
                        "receipt_path": "pre_market_proofs/RH-DIRECT-PARTIAL.json",
                        "decision_sha256": compute_regime_decision_sha256(
                            {
                                "record_id": "RH-DIRECT-PARTIAL",
                                "decision_date": day1,
                                "effective_date": day2,
                                "symbol": SECTOR,
                                "prev_regime": "TRANSITION",
                                "new_regime": "RANGE_GRID",
                                "record_type": "LIVE",
                            },
                            f"{day1}T16:00:00+08:00",
                        ),
                    },
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="tsa_token_b64"):
        SimulationEngine(partial_bundle, run_mode="formal").run()


def test_formal_resume_rejects_appended_modified_or_out_of_window_backfill():
    days = sessions(8)  # 2026-11-23..2026-12-02 (index 5 = 2026-11-30, index 6 = 2026-12-01)
    day_fri = days[4]  # 2026-11-27
    day_mon = days[5]  # 2026-11-30
    day_tue = days[6]  # 2026-12-01

    # 1. With forward_eval_start_date=None, formal runs to 2026-11-30 cleanly when no BACKFILL exists
    clean_bundle = make_bundle(days, initial_routes={SECTOR: "TRANSITION"}, forward_eval_start_date=None)
    formal = run_simulation(clean_bundle, end_date=day_mon, run_mode="formal")
    clean_state = json.loads(json.dumps(formal.export_state(), allow_nan=False))

    # Append a BACKFILL taking effect at <= checkpoint (effective_date=2026-11-26 inside window)
    appended_in_win_bundle = make_bundle(
        days,
        initial_routes={SECTOR: "TRANSITION"},
        forward_eval_start_date=None,
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-APP",
                    "decision_date": days[2],
                    "effective_date": days[3],  # 2026-11-26
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "post-hoc appended backfill",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="BACKFILL"):
        SimulationEngine.from_state(appended_in_win_bundle, clean_state, run_mode="formal").run(end_date=day_tue)
    # Incremental replay without recalculating prior history must also reject!
    with pytest.raises(ValueError, match="BACKFILL"):
        SimulationEngine.from_state(appended_in_win_bundle, clean_state, run_mode="replay").run(end_date=day_tue)

    # Append a BACKFILL taking effect at <= checkpoint outside 23..26 window
    tampered_bundle = make_bundle(
        days,
        initial_routes={SECTOR: "TRANSITION"},
        forward_eval_start_date=None,
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-LATE",
                    "decision_date": day_fri,
                    "effective_date": day_mon,
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "post-checkpoint backfill",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="BACKFILL"):
        SimulationEngine.from_state(tampered_bundle, clean_state, run_mode="formal").run(end_date=day_tue)
    with pytest.raises(ValueError, match="BACKFILL"):
        SimulationEngine.from_state(tampered_bundle, clean_state, run_mode="formal").step_day(day_tue)

    # Even if replay ran from scratch past backfill_replay_window, formal rejects out-of-window and null-window
    late_replay = run_simulation(tampered_bundle, end_date=day_mon, run_mode="replay")
    late_state = json.loads(json.dumps(late_replay.export_state(), allow_nan=False))
    with pytest.raises(ValueError, match="超出初始化回放区间"):
        SimulationEngine.from_state(tampered_bundle, late_state, run_mode="formal").run(end_date=day_tue)
    null_win_bundle = make_bundle(
        days,
        initial_routes={SECTOR: "TRANSITION"},
        forward_eval_start_date=None,
        backfill_replay_window=None,
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-LATE",
                    "decision_date": day_fri,
                    "effective_date": day_mon,
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "post-checkpoint backfill",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="backfill_replay_window"):
        SimulationEngine.from_state(null_win_bundle, late_state, run_mode="formal").run(end_date=day_tue)

    # 2. Valid replay in window (2026-11-23..2026-11-26) -> formal resume with forward_eval_start_date=None succeeds,
    #    and deleting or modifying any field (including decision_ref / trigger_alerts) fails the gate.
    in_window_bundle = make_bundle(
        days,
        initial_routes={SECTOR: "TRANSITION"},
        forward_eval_start_date=None,
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-INIT",
                    "decision_date": days[0],
                    "effective_date": days[1],
                    "prev_regime": "TRANSITION",
                    "regime": "TRANSITION",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "in-window init backfill",
                    "decision_ref": "2026_decisions.md#2026-11-23-510720.SH",
                    "trigger_alerts": ["[新触发] 20日跌破"],
                    "card_path": "project_content/radar_cards/510720.SH_红利国企.md",
                }
            ]
        },
    )
    init_engine = run_simulation(in_window_bundle, end_date=days[3], run_mode="replay")  # through 2026-11-26
    init_state = json.loads(json.dumps(init_engine.export_state(), allow_nan=False))
    resumed = SimulationEngine.from_state(in_window_bundle, init_state, run_mode="formal")
    summary = resumed.run(end_date=day_tue)
    assert summary["last_processed_date"] == day_tue

    # Deleting previously replayed BACKFILL must fail
    deleted_bundle = make_bundle(days, initial_routes={SECTOR: "TRANSITION"}, forward_eval_start_date=None)
    with pytest.raises(ValueError, match="BACKFILL"):
        SimulationEngine.from_state(deleted_bundle, init_state, run_mode="formal").run(end_date=day_tue)

    # Modifying metadata (decision_ref / trigger_alerts) on replayed BACKFILL must fail
    meta_modified_bundle = make_bundle(
        days,
        initial_routes={SECTOR: "TRANSITION"},
        forward_eval_start_date=None,
        record_overrides={
            SECTOR: [
                {
                    "record_id": "RH-BF-INIT",
                    "decision_date": days[0],
                    "effective_date": days[1],
                    "prev_regime": "TRANSITION",
                    "regime": "TRANSITION",
                    "record_type": "BACKFILL",
                    "recorded_at": None,
                    "time_evidence_boundary": "in-window init backfill",
                    "decision_ref": "2026_decisions.md#TAMPERED",
                    "trigger_alerts": ["[新触发] 篡改"],
                    "card_path": "project_content/radar_cards/510720.SH_红利国企.md",
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="不一致|BACKFILL"):
        SimulationEngine.from_state(meta_modified_bundle, init_state, run_mode="formal").run(end_date=day_tue)
