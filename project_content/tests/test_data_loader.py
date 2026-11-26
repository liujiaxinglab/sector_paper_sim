"""Unit tests for BS-01 calendar, corporate-actions, and route-history loading."""

from __future__ import annotations

import base64
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

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
    DEFAULT_SIM_CONFIG_PATH,
    TDX_NUMERIC_COLUMNS,
    TDX_REQUIRED_COLUMNS,
    RegimeHistory,
    build_forward_adjusted_daily,
    compute_live_attestation_payload,
    compute_regime_decision_sha256,
    find_latest_tdx_csv,
    load_corporate_actions_ledger,
    load_market_bundle,
    load_regime_history,
    load_sim_config,
    load_tdx_daily,
    load_trade_calendar,
    assess_corporate_actions_gate,
)

EXPECTED_SYMBOLS = [
    "159330.SZ",
    "512400.SH",
    "510720.SH",
    "159622.SZ",
    "515790.SH",
    "512480.SH",
    "512880.SH",
]
ISO_DAYS = ["2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26"]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_calendar(directory: Path, trading_dates: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = pd.date_range(trading_dates[0], trading_dates[-1], freq="D")
    path = directory / "20261126_100000.csv"
    path.write_text(
        "calendar_date,is_trading_day\n"
        + "".join(
            f"{day:%Y-%m-%d},{int(day.strftime('%Y-%m-%d') in trading_dates)}\n"
            for day in rows
        ),
        encoding="utf-8",
    )
    return path


def _write_tdx_batch(directory: Path, symbol: str, dates: list[str]) -> Path:
    symbol_dir = directory / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    path = symbol_dir / "20261126_100000.csv"
    path.write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        + "".join(
            f"{day},1.00,1.02,0.98,1.01,1000.0,1000,0\n" for day in dates
        ),
        encoding="utf-8",
    )
    return path


def _make_synthetic_config(tmp_path: Path, bar_dates: dict[str, list[str]]) -> dict:
    tdx_dir = tmp_path / "tdx"
    symbols = list(bar_dates)
    sectors = symbols[1:]
    for symbol, dates in bar_dates.items():
        _write_tdx_batch(tdx_dir, symbol, dates)
    calendar_dir = tmp_path / "calendar"
    _write_calendar(calendar_dir, ISO_DAYS)

    watchlist_file = tmp_path / "watchlist.json"
    _write_json(
        watchlist_file,
        {
            "universe": [
                {
                    "symbol": symbol,
                    "name": symbol,
                    "role": "benchmark" if index == 0 else "sector",
                    "current_regime": "BENCHMARK" if index == 0 else "UNRATED",
                }
                for index, symbol in enumerate(symbols)
            ]
        },
    )
    ledger_file = tmp_path / "corporate_actions_ledger.json"
    _write_json(
        ledger_file,
        {
            "schema_version": "2.0.0",
            "as_of": "2026-11-26",
            "cash_dividends_by_symbol": {symbol: [] for symbol in symbols},
            "splits_by_symbol": {symbol: [] for symbol in symbols},
        },
    )
    regime_file = tmp_path / "regime_history.json"
    _write_json(
        regime_file,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {symbol: "UNRATED" for symbol in sectors},
            "records": [],
        },
    )
    return {
        "project_name": "sector_paper_sim_test",
        "start_date": "2026-11-23",
        "upstream_paths": {
            "tdx_daily_dir": str(tdx_dir),
            "trade_calendar_dir": str(calendar_dir),
            "corporate_actions_ledger_file": str(ledger_file),
            "regime_history_file": str(regime_file),
            "watchlist_file": str(watchlist_file),
        },
        "account_defaults": {"initial_capital": 100000.0},
        "strategies": {},
        "universe": {"benchmark": symbols[0], "sectors": sectors},
    }


def test_config_is_relocated_and_uses_the_v2_contract() -> None:
    config = load_sim_config(DEFAULT_SIM_CONFIG_PATH)
    assert DEFAULT_SIM_CONFIG_PATH == (
        Path(__file__).resolve().parents[1] / "config" / "sim_config.json"
    )
    assert config["start_date"] == "2026-11-23"
    assert set(config["upstream_paths"]) == {
        "tdx_daily_dir",
        "trade_calendar_dir",
        "corporate_actions_ledger_file",
        "regime_history_file",
        "watchlist_file",
    }
    assert config["strategies"]["grid"] == {
        "step_mode": "atr20_locked",
        "atr_window": 20,
        "min_step_pct": 0.015,
        "grid_levels_each_side": 4,
        "total_slices": 8,
        "initial_slices": 4,
        "initial_position_ratio": 0.5,
    }
    assert "risk_fraction" not in config["strategies"]["turtle"]


def test_real_market_bundle_uses_bs01_calendar_and_ledger() -> None:
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)

    assert bundle.benchmark == EXPECTED_SYMBOLS[0]
    assert set(bundle.sectors) == set(EXPECTED_SYMBOLS[1:])
    assert set(bundle.symbols) == set(EXPECTED_SYMBOLS)
    assert set(bundle.bars) == set(EXPECTED_SYMBOLS)
    assert bundle.metadata["calendar_file"].endswith("20261126_100336.csv")
    assert bundle.metadata["corporate_actions_as_of"] == "2026-11-26"
    assert bundle.regime_history.anchor_start_date == "2026-11-23"
    assert bundle.authoritative_trading_dates == sorted(
        bundle.authoritative_trading_dates
    )
    assert "2026-11-23" in bundle.authoritative_trading_dates
    assert bundle.common_trading_dates[-1] == bundle.common_end_date

    for symbol in EXPECTED_SYMBOLS:
        bars = bundle.bars[symbol]
        assert list(bars.columns) == list(TDX_REQUIRED_COLUMNS)
        assert bars["date"].tolist() == sorted(set(bars["date"].tolist()))
        for column in TDX_NUMERIC_COLUMNS:
            assert pd.api.types.is_float_dtype(bars[column])
        assert symbol in bundle.splits
        assert symbol in bundle.dividends
        assert bundle.adjusted_bars[symbol]["adj_factor"].iloc[-1] == pytest.approx(1.0)

    assert bundle.regime_history.regime_for("510720.SH", "2026-11-23") == "TRANSITION"


def test_bs01_calendar_keeps_market_day_when_one_symbol_has_no_bar(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {
            "159330.SZ": ["2026-11-23", "2026-11-25", "2026-11-26"],
            "512400.SH": ISO_DAYS,
        },
    )
    bundle = load_market_bundle(config)

    assert bundle.common_trading_dates == ISO_DAYS
    assert bundle.authoritative_trading_dates == ISO_DAYS
    assert bundle.suspension_dates_by_symbol == {"159330.SZ": ["2026-11-24"]}
    assert bundle.metadata["missing_trading_dates"]["159330.SZ"] == ["2026-11-24"]
    assert "2026-11-24" not in bundle.bars["159330.SZ"]["date"].tolist()
    assert "2026-11-24" in bundle.bars["512400.SH"]["date"].tolist()


def test_latest_csv_lag_is_separate_from_in_range_suspension(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {
            "159330.SZ": ISO_DAYS,
            "512400.SH": ISO_DAYS[:-1],
        },
    )
    bundle = load_market_bundle(config)

    assert bundle.common_end_date == "2026-11-25"
    assert bundle.common_trading_dates == ISO_DAYS[:-1]
    assert bundle.lag_warnings[0]["symbol"] == "512400.SH"
    assert bundle.lag_warnings[0]["lag_trading_days"] == 1
    assert bundle.lagged_dates_by_symbol["512400.SH"] == ["2026-11-26"]
    assert "512400.SH" not in bundle.suspension_dates_by_symbol


def test_corporate_actions_ledger_sorts_events_and_validates_schema(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    _write_json(
        ledger_path,
        {
            "schema_version": "2.0.0",
            "as_of": "2026-11-26",
            "cash_dividends_by_symbol": {
                "510720.SH": [
                    {
                        "ex_date": "2026-11-25", "cash_per_unit": 0.02,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/dividend.pdf", "source_pdf_sha256": "a" * 64,
                    },
                    {
                        "ex_date": "2026-11-23", "cash_per_unit": 0.01,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/dividend.pdf", "source_pdf_sha256": "a" * 64,
                    },
                ]
            },
            "splits_by_symbol": {
                "512480.SH": [
                    {
                        "effective_date": "2026-11-25", "factor": 0.5,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": "b" * 64,
                    },
                    {
                        "effective_date": "2026-11-23", "factor": 0.8,
                        "source_class": "OFFICIAL_TEST", "event_status": "VERIFIED",
                        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": "b" * 64,
                    },
                ]
            },
        },
    )
    ledger = load_corporate_actions_ledger(ledger_path)
    assert ledger.as_of == "2026-11-26"
    assert [event["ex_date"] for event in ledger.cash_dividends_by_symbol["510720.SH"]] == [
        "2026-11-23", "2026-11-25"
    ]
    assert [event["effective_date"] for event in ledger.splits_by_symbol["512480.SH"]] == [
        "2026-11-23", "2026-11-25"
    ]
    assert all(
        event["event_status"] == "VERIFIED"
        and event["source_class"] == "OFFICIAL_TEST"
        and event["source_pdf_sha256"] == "a" * 64
        for event in ledger.cash_dividends_by_symbol["510720.SH"]
    )

    invalid_path = tmp_path / "invalid_ledger.json"
    _write_json(invalid_path, {"schema_version": "1.0.0", "as_of": "2026-11-26"})
    with pytest.raises(ValueError, match="schema_version"):
        load_corporate_actions_ledger(invalid_path)


def test_provisional_actions_are_retained_but_never_applied(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {"159330.SZ": ISO_DAYS, "512400.SH": ISO_DAYS},
    )
    ledger_path = Path(config["upstream_paths"]["corporate_actions_ledger_file"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["splits_by_symbol"]["512400.SH"] = [
        {
            "effective_date": "2026-11-25", "factor": 0.5,
            "source_class": "PROVISIONAL_TEST", "event_status": "PROVISIONAL",
            "source_pdf": None, "source_pdf_sha256": None,
        }
    ]
    _write_json(ledger_path, payload)

    bundle = load_market_bundle(config)
    assert bundle.splits["512400.SH"] == []
    assert bundle.adjusted_bars["512400.SH"]["adj_factor"].tolist() == [1.0] * len(ISO_DAYS)
    pending = bundle.metadata["unverified_corporate_action_events"]
    assert len(pending) == 1
    assert pending[0]["event_status"] == "PROVISIONAL"
    assert pending[0]["source_class"] == "PROVISIONAL_TEST"
    assert pending[0]["date"] == "2026-11-25"
    assert "缺少官方 PDF" in pending[0]["evidence_gap"]


def test_provisional_pdf_and_hash_must_be_both_null_or_a_valid_pair(tmp_path: Path) -> None:
    config = _make_synthetic_config(
        tmp_path,
        {"159330.SZ": ISO_DAYS, "512400.SH": ISO_DAYS},
    )
    ledger_path = Path(config["upstream_paths"]["corporate_actions_ledger_file"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["splits_by_symbol"]["512400.SH"] = [{
        "effective_date": "2026-11-25", "factor": 0.5,
        "source_class": "PROVISIONAL_TEST", "event_status": "PROVISIONAL",
        "source_pdf": "raw/test/split.pdf", "source_pdf_sha256": None,
    }]
    _write_json(ledger_path, payload)

    with pytest.raises(ValueError, match="PROVISIONAL event source_pdf and source_pdf_sha256"):
        load_corporate_actions_ledger(ledger_path)


def test_shared_gate_blocks_lag_and_effective_provisional_events() -> None:
    future = [{"date": "2026-11-27", "symbol": "512400.SH", "event_status": "PROVISIONAL"}]
    assert assess_corporate_actions_gate("2026-11-26", "2026-11-26", future)["allowed"]
    lag = assess_corporate_actions_gate("2026-11-25", "2026-11-26", [])
    assert not lag["allowed"]
    effective = assess_corporate_actions_gate("2026-11-26", "2026-11-26", future[:0] + [
        {"date": "2026-11-26", "symbol": "512400.SH", "event_status": "PROVISIONAL"}
    ])
    assert not effective["allowed"]
    assert effective["unverified_events"][0]["symbol"] == "512400.SH"


def test_real_512480_split_adjustment_comes_from_the_ledger() -> None:
    bundle = load_market_bundle(DEFAULT_SIM_CONFIG_PATH)
    split_events = bundle.splits["512480.SH"]
    assert any(
        event["effective_date"] == "2026-07-03" and event["factor"] == 0.5
        for event in split_events
    )

    raw = bundle.bars["512480.SH"].set_index("date")
    adjusted = bundle.adjusted_bars["512480.SH"].set_index("date")
    assert raw.loc["2026-07-02", "close"] == pytest.approx(2.700)
    assert adjusted.loc["2026-07-02", "adj_factor"] == pytest.approx(0.5)
    assert adjusted.loc["2026-07-02", "close"] == pytest.approx(1.350)
    assert raw.loc["2026-07-03", "close"] == pytest.approx(1.331)
    assert adjusted.loc["2026-07-03", "adj_factor"] == pytest.approx(1.0)


class _TestTSA:
    """Test-only offline RFC 3161 TSA issuer."""

    def __init__(self, cn: str = "PaperSim Test TSA") -> None:
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


def _attach_live_proof(
    journal_dir: Path,
    record: dict,
    *,
    tsa: _TestTSA | None = None,
    gen_time: datetime | None = None,
    entry_sha256: str | None = None,
) -> Path:
    recorded_cst = datetime.fromisoformat(record["recorded_at"].replace("Z", "+00:00")).astimezone(CHINA_TZ).isoformat()
    decision_sha = compute_regime_decision_sha256(record, recorded_cst)
    rel_path = f"pre_market_proofs/{record['record_id']}.json"
    receipt_path = journal_dir / rel_path
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_bytes = compute_live_attestation_payload(
        record_id=record["record_id"],
        decision_date=record["decision_date"],
        effective_date=record["effective_date"],
        symbol=record["symbol"],
        prev_regime=record["prev_regime"],
        new_regime=record["new_regime"],
        recorded_at_cst=recorded_cst,
        decision_sha256=decision_sha,
        decision_entry_sha256=entry_sha256,
        decision_ref=record.get("decision_ref"),
        card_path=record.get("card_path"),
    )
    receipt = {
        "schema_version": "2.0.0",
        "record_id": record["record_id"],
        "decision_date": record["decision_date"],
        "effective_date": record["effective_date"],
        "symbol": record["symbol"],
        "prev_regime": record["prev_regime"],
        "new_regime": record["new_regime"],
        "record_type": "LIVE",
        "recorded_at": record["recorded_at"],
        "recorded_at_cst": recorded_cst,
        "decision_sha256": decision_sha,
        "decision_entry_sha256": entry_sha256,
        "attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
    }
    _write_json(receipt_path, receipt)
    proof: dict = {
        "receipt_path": rel_path,
        "decision_sha256": decision_sha,
        "decision_entry_sha256": entry_sha256,
    }
    if tsa is not None:
        ts_dt = gen_time if gen_time is not None else datetime.fromisoformat(recorded_cst)
        tsr_bytes = tsa.sign_digest(hashlib.sha256(attestation_bytes).digest(), ts_dt)
        tsr_rel = f"pre_market_proofs/{record['record_id']}.tsr"
        (journal_dir / tsr_rel).write_bytes(tsr_bytes)
        proof["tsa_token_path"] = tsr_rel
        proof["tsa_token_b64"] = base64.b64encode(tsr_bytes).decode("ascii")
    record["pre_market_proof"] = proof
    return receipt_path


def test_regime_history_uses_effective_date_and_maps_unrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tsa = _TestTSA()
    monkeypatch.setattr("src.data_loader._load_default_trusted_tsa_roots", lambda: (tsa.root_cert,))
    history_path = tmp_path / "regime_history.json"
    rec_a = {
        "record_id": "RH-20261125-001",
        "decision_date": "2026-11-25",
        "effective_date": "2026-11-26",
        "symbol": "512400.SH",
        "prev_regime": "TREND_UP",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-25T16:00:00+08:00",
    }
    rec_b = {
        "record_id": "RH-20261123-001",
        "decision_date": "2026-11-23",
        "effective_date": "2026-11-24",
        "symbol": "512400.SH",
        "prev_regime": "TRANSITION",
        "new_regime": "TREND_UP",
        "record_type": "LIVE",
        "recorded_at": "2026-11-23T16:00:00+08:00",
    }
    _attach_live_proof(tmp_path, rec_a, tsa=tsa, gen_time=datetime(2026, 11, 25, 16, 0, tzinfo=CHINA_TZ))
    _attach_live_proof(tmp_path, rec_b, tsa=tsa, gen_time=datetime(2026, 11, 23, 16, 0, tzinfo=CHINA_TZ))
    _write_json(
        history_path,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"512400.SH": "UNRATED"},
            "records": [rec_a, rec_b],
        },
    )
    history = load_regime_history(history_path, trading_calendar=ISO_DAYS)
    assert history.regime_for("512400.SH", "2026-11-23") == "TRANSITION"
    assert history.regime_for("512400.SH", "2026-11-24") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-25") == "TREND_UP"
    assert history.regime_for("512400.SH", "2026-11-26") == "RANGE_GRID"
    with pytest.raises(ValueError, match="precedes"):
        history.regime_for("512400.SH", "2026-11-22")


def test_regime_history_enforces_bs01_t1_and_live_backfill_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsa = _TestTSA("Trusted Sim TSA")
    untrusted_tsa = _TestTSA("Untrusted Sim TSA")
    monkeypatch.setattr("src.data_loader._load_default_trusted_tsa_roots", lambda: (tsa.root_cert,))
    calendar = ["2026-11-26", "2026-11-27", "2026-11-30", "2026-12-01"]
    base = {
        "schema_version": "2.0.0",
        "anchor_start_date": "2026-11-26",
        "initial_regimes": {"512400.SH": "TRANSITION"},
        "records": [
            {
                "record_id": "RH-20261127-001",
                "decision_date": "2026-11-27",
                "effective_date": "2026-11-30",
                "symbol": "512400.SH",
                "prev_regime": "TRANSITION",
                "new_regime": "RANGE_GRID",
                "record_type": "LIVE",
                "recorded_at": "2026-11-30T01:29:59Z",
            }
        ],
    }
    pre_open_ts = datetime(2026, 11, 30, 9, 29, 59, tzinfo=CHINA_TZ)
    receipt_path = _attach_live_proof(tmp_path, base["records"][0], tsa=tsa, gen_time=pre_open_ts)
    tsr_path = tmp_path / "pre_market_proofs" / "RH-20261127-001.tsr"
    attestation_digest = bytes.fromhex(json.loads(receipt_path.read_text(encoding="utf-8"))["attestation_sha256"])
    history_path = tmp_path / "regime_history.json"
    _write_json(history_path, base)
    loaded = load_regime_history(history_path, trading_calendar=calendar)
    assert loaded.regime_for("512400.SH", "2026-11-27") == "TRANSITION"
    assert loaded.regime_for("512400.SH", "2026-11-30") == "RANGE_GRID"

    # Even with post-market file mtime (re-clone), valid pre-market TSA passes
    post_mtime = datetime(2026, 11, 30, 20, 48, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(receipt_path, (post_mtime, post_mtime))
    os.utime(tsr_path, (post_mtime, post_mtime))
    assert load_regime_history(history_path, trading_calendar=calendar).regime_for("512400.SH", "2026-11-30") == "RANGE_GRID"

    # Calendar +1 day (Saturday 2026-11-28) instead of BS-01 Monday 2026-11-30 must fail
    bad_weekend = json.loads(json.dumps(base))
    bad_weekend["records"][0]["effective_date"] = "2026-11-28"
    _write_json(history_path, bad_weekend)
    with pytest.raises(ValueError, match="BS-01 T\\+1"):
        load_regime_history(history_path, trading_calendar=calendar)

    for bad_mut, pattern in [
        (lambda d: d.__setitem__("schema_version", "1.0.0"), "schema_version"),
        (lambda d: d["records"][0].__setitem__("record_type", "OTHER"), "record_type"),
        (lambda d: d["records"][0].__setitem__("recorded_at", None), "recorded_at"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00+07:60"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00+24:00"), "ISO 8601"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-27T16:00:00-00:00"), "'-00:00'"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-26T23:59:59+08:00"), "precedes decision_date"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-30T09:30:00+08:00"), "market open"),
        (lambda d: d["records"][0].__setitem__("recorded_at", "2026-11-30T01:30:00Z"), "market open"),
        (lambda d: d["records"][0].pop("pre_market_proof"), "pre_market_proof"),
        (lambda d: d["records"][0]["pre_market_proof"].pop("tsa_token_b64"), "tsa_token_b64"),
        (lambda d: d["records"][0]["pre_market_proof"].__setitem__("decision_sha256", "0" * 64), "does not match"),
        (lambda d: d["records"][0]["pre_market_proof"].__setitem__("receipt_path", "pre_market_proofs/MISSING.json"), "Missing LIVE pre_market_proof"),
        (lambda d: d["records"][0].__setitem__("new_regime", "TREND_UP"), "does not match"),
    ]:
        bad = json.loads(json.dumps(base))
        bad_mut(bad)
        _write_json(history_path, bad)
        with pytest.raises(ValueError, match=pattern):
            load_regime_history(history_path, trading_calendar=calendar)

    # Untrusted CA TSA token with rolled-back mtime must fail
    untrusted_data = json.loads(json.dumps(base))
    untrusted_tsr = untrusted_tsa.sign_digest(attestation_digest, pre_open_ts)
    tsr_path.write_bytes(untrusted_tsr)
    untrusted_data["records"][0]["pre_market_proof"]["tsa_token_b64"] = base64.b64encode(untrusted_tsr).decode("ascii")
    os.utime(receipt_path, (pre_open_ts.timestamp(), pre_open_ts.timestamp()))
    os.utime(tsr_path, (pre_open_ts.timestamp(), pre_open_ts.timestamp()))
    _write_json(history_path, untrusted_data)
    with pytest.raises(ValueError, match="unverified source"):
        load_regime_history(history_path, trading_calendar=calendar)

    # Post-market TSA reception time (even with rolled-back pre-market mtime) must fail
    for bad_ts in (datetime(2026, 11, 30, 9, 30, 0, tzinfo=CHINA_TZ), datetime(2026, 11, 30, 20, 48, 0, tzinfo=CHINA_TZ)):
        late_tsr = tsa.sign_digest(attestation_digest, bad_ts)
        tsr_path.write_bytes(late_tsr)
        late_data = json.loads(json.dumps(base))
        late_data["records"][0]["pre_market_proof"]["tsa_token_b64"] = base64.b64encode(late_tsr).decode("ascii")
        os.utime(receipt_path, (pre_open_ts.timestamp(), pre_open_ts.timestamp()))
        os.utime(tsr_path, (pre_open_ts.timestamp(), pre_open_ts.timestamp()))
        _write_json(history_path, late_data)
        with pytest.raises(ValueError, match="market open"):
            load_regime_history(history_path, trading_calendar=calendar)

    # BACKFILL requires time_evidence_boundary when recorded_at is null
    backfill = json.loads(json.dumps(base))
    backfill["records"][0]["record_type"] = "BACKFILL"
    backfill["records"][0]["recorded_at"] = None
    backfill["records"][0].pop("pre_market_proof", None)
    _write_json(history_path, backfill)
    with pytest.raises(ValueError, match="time_evidence_boundary"):
        load_regime_history(history_path, trading_calendar=calendar)
    backfill["records"][0]["time_evidence_boundary"] = "backfill committed at 2026-11-30T10:30:00+08:00"
    _write_json(history_path, backfill)
    bf_loaded = load_regime_history(history_path, trading_calendar=calendar)
    with pytest.raises(ValueError, match="BACKFILL"):
        bf_loaded.validate_for_run_mode("formal", ["2026-11-27", "2026-11-30"])
    bf_loaded.validate_for_run_mode("replay", ["2026-11-27", "2026-11-30"])


def test_regime_history_rejects_backfilled_git_dates_and_in_memory_missing_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data_loader import _extract_decision_entry_block

    tsa = _TestTSA("Trusted Git Test TSA")
    repo = tmp_path / "radar_git"
    journal = repo / "project_content" / "journal"
    journal.mkdir(parents=True)
    record = {
        "record_id": "RH-20261123-001",
        "decision_date": "2026-11-23",
        "effective_date": "2026-11-24",
        "symbol": "510720.SH",
        "prev_regime": "TRANSITION",
        "new_regime": "RANGE_GRID",
        "record_type": "LIVE",
        "recorded_at": "2026-11-23T16:00:00+08:00",
        "decision_ref": "2026_decisions.md#2026-11-23-510720.SH",
    }
    decisions = journal / "2026_decisions.md"
    decisions.write_text('<a id="2026-11-23-510720.SH"></a>\n\n### 510720.SH\n- RANGE_GRID\n', encoding="utf-8")
    block = _extract_decision_entry_block(decisions.read_text(encoding="utf-8"), "2026-11-23-510720.SH")
    assert block is not None
    entry_sha = hashlib.sha256(block.encode("utf-8")).hexdigest()

    # Unsigned receipt with backfilled st_mtime and backfilled local Git commit must fail
    receipt_path = _attach_live_proof(journal, record, tsa=None, entry_sha256=entry_sha)
    pre_ts = datetime(2026, 11, 23, 16, 0, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(receipt_path, (pre_ts, pre_ts))
    history_path = journal / "regime_history.json"
    _write_json(
        history_path,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"510720.SH": "TRANSITION"},
            "records": [record],
        },
    )

    def _git(args: list[str], commit_ts: str | None = None) -> None:
        env = {**os.environ}
        if commit_ts:
            env["GIT_AUTHOR_DATE"] = commit_ts
            env["GIT_COMMITTER_DATE"] = commit_ts
        proc = subprocess.run(["git", "-C", str(repo), *args], env=env, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr

    _git(["init"])
    _git(["config", "user.name", "Tester"])
    _git(["config", "user.email", "tester@example.com"])
    _git(["add", "."])
    _git(["commit", "-m", "backfilled pre-market git date"], commit_ts="2026-11-23T16:00:00+08:00")
    with pytest.raises(ValueError, match="tsa_token_b64"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)

    # Self-signed TSA token committed with pre-market Git dates must also fail under production roots
    _attach_live_proof(
        journal, record, tsa=tsa, gen_time=datetime(2026, 11, 23, 16, 5, 0, tzinfo=CHINA_TZ), entry_sha256=entry_sha
    )
    _write_json(
        history_path,
        {
            "schema_version": "2.0.0",
            "anchor_start_date": "2026-11-23",
            "initial_regimes": {"510720.SH": "TRANSITION"},
            "records": [record],
        },
    )
    _git(["add", "."])
    _git(["commit", "-m", "untrusted tsr"], commit_ts="2026-11-23T16:05:00+08:00")
    with pytest.raises(ValueError, match="unverified source"):
        load_regime_history(history_path, trading_calendar=ISO_DAYS)

    # In-memory RegimeHistory (source_path=None) must NOT skip pre_market_proof verification
    in_memory_unproven = RegimeHistory(
        schema_version="2.0.0",
        anchor_start_date="2026-11-23",
        initial_regimes={"510720.SH": "TRANSITION"},
        records_by_symbol={
            "510720.SH": [
                {
                    "record_id": "RH-MEM-001",
                    "decision_date": "2026-11-23",
                    "effective_date": "2026-11-24",
                    "prev_regime": "TRANSITION",
                    "regime": "RANGE_GRID",
                    "record_type": "LIVE",
                    "recorded_at": "2026-11-23T16:00:00+08:00",
                }
            ]
        },
        source_path=None,
    )
    with pytest.raises(ValueError, match="pre_market_proof"):
        in_memory_unproven.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])

    # When TSA root is trusted, valid pre-market proof passes on both disk (with post-market mtime) and in-memory
    monkeypatch.setattr("src.data_loader._load_default_trusted_tsa_roots", lambda: (tsa.root_cert,))
    post_ts = datetime(2026, 11, 26, 20, 48, 0, tzinfo=CHINA_TZ).timestamp()
    os.utime(receipt_path, (post_ts, post_ts))
    loaded = load_regime_history(history_path, trading_calendar=ISO_DAYS)
    assert loaded.regime_for("510720.SH", "2026-11-24") == "RANGE_GRID"
    in_memory_valid = RegimeHistory(
        schema_version="2.0.0",
        anchor_start_date="2026-11-23",
        initial_regimes={"510720.SH": "TRANSITION"},
        records_by_symbol={
            "510720.SH": [
                {
                    **record,
                    "regime": "RANGE_GRID",
                }
            ]
        },
        source_path=None,
    )
    in_memory_valid.validate_for_run_mode("formal", ["2026-11-23", "2026-11-24"])


def test_tdx_latest_batch_deduplicates_and_validates_dates(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "510720.SH"
    symbol_dir.mkdir()
    (symbol_dir / "20261120_093000.csv").write_text(
        "date,open,high,low,close,amount,volume\n"
        "2026-11-20,1,1,1,1,100,100\n",
        encoding="utf-8",
    )
    latest = symbol_dir / "20261123_151557.csv"
    latest.write_text(
        "date,open,high,low,close,amount,volume,reserved\n"
        "20261120,1.2,1.22,1.19,1.21,300000,250000,0\n"
        "2026/11/18,1.1,1.12,1.09,1.11,100000,90000,0\n"
        "2026-11-18,1.105,1.125,1.095,1.115,150000,135000,0\n",
        encoding="utf-8",
    )
    assert find_latest_tdx_csv(tmp_path, "510720.SH") == latest
    frame = load_tdx_daily(tmp_path, "510720.SH")
    assert frame["date"].tolist() == ["2026-11-18", "2026-11-20"]
    assert frame["close"].tolist() == pytest.approx([1.115, 1.21])
    assert list(frame.columns) == list(TDX_REQUIRED_COLUMNS)


@pytest.mark.parametrize(
    "bad_row",
    [
        "2026-11-20,,1.02,0.99,1.01,1000,1000",
        "not-a-date,1.00,1.02,0.99,1.01,1000,1000",
        "2026-11-20,1.00,1.02,0.99,-1.01,1000,1000",
    ],
)
def test_tdx_loader_rejects_corrupt_rows(tmp_path: Path, bad_row: str) -> None:
    symbol_dir = tmp_path / "159330.SZ"
    symbol_dir.mkdir()
    (symbol_dir / "20261123_120000.csv").write_text(
        "date,open,high,low,close,amount,volume\n" + bad_row + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_tdx_daily(tmp_path, "159330.SZ")


def test_forward_adjustment_applies_split_and_dividend_without_mutation() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06"],
            "open": [2.00, 2.00, 1.00, 0.95],
            "high": [2.04, 2.02, 1.02, 0.98],
            "low": [1.98, 1.98, 0.98, 0.94],
            "close": [2.00, 2.00, 1.00, 0.96],
            "amount": [1000.0] * 4,
            "volume": [500.0, 500.0, 1000.0, 1050.0],
        }
    )
    before = raw.copy(deep=True)
    adjusted = build_forward_adjusted_daily(
        raw,
        split_events=[{"effective_date": "2026-07-03", "factor": 0.5}],
        dividend_events={"2026-07-06": 0.05},
    )
    pd.testing.assert_frame_equal(raw, before)
    assert adjusted["adj_factor"].tolist() == pytest.approx([0.475, 0.475, 0.95, 1.0])
    assert adjusted["close"].tolist() == pytest.approx([0.95, 0.95, 0.95, 0.96])


def test_bs01_calendar_loader_filters_trading_days(tmp_path: Path) -> None:
    calendar_dir = tmp_path / "calendar"
    _write_calendar(calendar_dir, ["2026-11-23", "2026-11-25"])
    calendar = load_trade_calendar(
        calendar_dir, start_date="2026-11-23", end_date="2026-11-25"
    )
    assert calendar == ["2026-11-23", "2026-11-25"]


def test_pytest_config_imports_package_from_an_external_working_directory(
    tmp_path: Path,
) -> None:
    repo_root = DEFAULT_SIM_CONFIG_PATH.parents[2]
    pyproject = repo_root / "pyproject.toml"
    probe_file = tmp_path / "test_external_import_probe.py"
    probe_file.write_text(
        "from importlib import import_module\n"
        "from src.data_loader import DEFAULT_SIM_CONFIG_PATH\n"
        "def test_package_import_resolves():\n"
        "    assert DEFAULT_SIM_CONFIG_PATH.is_file()\n",
        encoding="utf-8",
    )
    probe_file.write_text(
        probe_file.read_text(encoding="utf-8")
        + "    for module in ('src.engine', 'src.strategies', 'src.reporter', 'src.run_daily_sim'):\n"
        + "        assert import_module(module)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "C:/Users/jxing/.conda/envs/quant/python.exe",
            "-m",
            "pytest",
            "-c",
            str(pyproject),
            "--rootdir",
            str(repo_root),
            str(probe_file),
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "1 passed" in result.stdout
