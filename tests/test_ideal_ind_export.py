"""
IdealQuant - IdealData Export Calibration
Compares IdealData export (ideal_ind_export.csv) with Python indicators.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.indicators import (
    SMA, EMA, DEMA, TEMA, HullMA, FRAMA, KAMA,
    RSI, Momentum, Qstick, RVI,
    ATR, ADX, HHV, LLV,
    MoneyFlowIndex,
    ChaikinMoneyFlow,
    EaseOfMovement,
    KeltnerUp, KeltnerDown,
    EnvelopeUp, EnvelopeDown, EnvelopeMid,
    KlingerOsc, MassIndex,
    PriceChannelUp, PriceChannelDown,
    LinearReg, LinearRegSlope,
    ChaikinVolatility,
    QQEF,
)
from src.indicators.oscillators import (
    CCI, MACD, WilliamsR, ROC, ChandeMomentum,
    StochasticFast, StochasticSlow,
    ElliotWaveOscillator, TRIX
)
from src.indicators.core import Stochastic
from src.indicators.trend import (
    DirectionalIndicatorPlus, DirectionalIndicatorMinus,
    AroonUp, AroonDown, AroonOsc,
    ParabolicSAR,
)
from src.indicators.volatility import BollingerUp, BollingerMid, BollingerDown, BollingerWidth
from src.indicators.volume import OBV, PVT, ChaikinOsc


def compare_indicator(name, ideal_values, py_values, tol_pct=1.0, tol_abs=0.01):
    ideal_values = np.asarray(ideal_values, dtype=float)
    py_values = np.asarray(py_values, dtype=float)

    min_len = min(len(ideal_values), len(py_values))
    ideal_values = ideal_values[:min_len]
    py_values = py_values[:min_len]

    mask = ~(np.isnan(ideal_values) | np.isnan(py_values))
    ideal_clean = ideal_values[mask]
    py_clean = py_values[mask]

    if ideal_clean.size == 0:
        return {
            "name": name,
            "status": "ERROR",
            "max_pct_diff": np.nan,
            "mean_pct_diff": np.nan,
            "max_abs_diff": np.nan,
            "mean_abs_diff": np.nan,
            "sample_count": 0,
            "is_match": False,
        }

    abs_diff = np.abs(ideal_clean - py_clean)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(np.abs(ideal_clean) > 1e-9, abs_diff / np.abs(ideal_clean) * 100, 0.0)

    max_pct_diff = float(np.nanmax(pct_diff))
    mean_pct_diff = float(np.nanmean(pct_diff))
    max_abs_diff = float(np.nanmax(abs_diff))
    mean_abs_diff = float(np.nanmean(abs_diff))

    is_match = (max_pct_diff <= tol_pct) or (max_abs_diff <= tol_abs)

    return {
        "name": name,
        "status": "OK" if is_match else "FAIL",
        "max_pct_diff": max_pct_diff,
        "mean_pct_diff": mean_pct_diff,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "sample_count": int(ideal_clean.size),
        "is_match": is_match,
    }


def run():
    export_path = PROJECT_ROOT / "data" / "ideal_ind_export.csv"
    raw_path = PROJECT_ROOT / "data" / "VIP_X030T_1dk_.csv"

    if not export_path.exists():
        print(f"[HATA] Export dosyası yok: {export_path}")
        return
    if not raw_path.exists():
        print(f"[HATA] Ham CSV yok: {raw_path}")
        return

    ideal_df = pd.read_csv(export_path, sep=";", decimal=".")
    raw_df = pd.read_csv(raw_path, sep=";", decimal=",", encoding="cp1254")
    raw_df.columns = ["Tarih", "Saat", "Acilis", "Yuksek", "Dusuk", "Kapanis", "Ortalama", "Hacim", "Lot"]

    raw_df["Time_Short"] = raw_df["Saat"].str[:5]
    raw_df["DateTime"] = raw_df["Tarih"] + " " + raw_df["Time_Short"]
    ideal_df["DateTime"] = ideal_df["Date"] + " " + ideal_df["Time"]

    raw_dt_to_idx = {dt: i for i, dt in enumerate(raw_df["DateTime"])}
    aligned_indices = np.array([raw_dt_to_idx.get(dt, -1) for dt in ideal_df["DateTime"]])
    valid_mask = aligned_indices >= 0

    if valid_mask.sum() == 0:
        print("[HATA] Tarih-saat eşleşmesi yok.")
        return

    idx = aligned_indices[valid_mask]

    o = raw_df["Acilis"].values.astype(float)
    h = raw_df["Yuksek"].values.astype(float)
    l = raw_df["Dusuk"].values.astype(float)
    c = raw_df["Kapanis"].values.astype(float)
    v = raw_df["Hacim"].values.astype(float)

    # Python indicators (full series)
    py_sma20 = np.array(SMA(c.tolist(), 20))
    py_ema20 = np.array(EMA(c.tolist(), 20))
    py_dema20 = np.array(DEMA(c.tolist(), 20))
    py_tema20 = np.array(TEMA(c.tolist(), 20))
    py_hull20 = np.array(HullMA(c.tolist(), 20))
    py_frama = np.array(FRAMA(c.tolist()))
    py_kama = np.array(KAMA(c.tolist(), 10, 2, 30))

    py_rsi14 = np.array(RSI(c.tolist(), 14))
    py_cci20 = np.array(CCI(h.tolist(), l.tolist(), c.tolist(), 20))
    py_mom10 = np.array(Momentum(c.tolist(), 10))
    py_roc10 = np.array(ROC(c.tolist(), 10))
    py_cmo9 = np.array(ChandeMomentum(c.tolist(), 9))
    py_willr14 = np.array(WilliamsR(h.tolist(), l.tolist(), c.tolist(), 14))
    py_stoch_fast = np.array(StochasticFast(h.tolist(), l.tolist(), c.tolist(), 14))
    py_stoch_slow = np.array(StochasticSlow(h.tolist(), l.tolist(), c.tolist(), 14, 3))
    py_qstick10 = np.array(Qstick(o.tolist(), c.tolist(), 10))
    py_rvi, py_rvi_sig = RVI(o.tolist(), h.tolist(), l.tolist(), c.tolist(), 10)
    py_rvi10 = np.array(py_rvi)
    py_rvi10_sig = np.array(py_rvi_sig)

    py_macd_line, _, _ = MACD(c.tolist(), 12, 26, 9)
    py_macd1226 = np.array(py_macd_line)

    py_hhv20 = np.array(HHV(h.tolist(), 20))
    py_llv20 = np.array(LLV(l.tolist(), 20))
    py_atr14 = np.array(ATR(h.tolist(), l.tolist(), c.tolist(), 14))
    py_adx14 = np.array(ADX(h.tolist(), l.tolist(), c.tolist(), 14))
    py_di_plus = np.array(DirectionalIndicatorPlus(h.tolist(), l.tolist(), c.tolist(), 14))
    py_di_minus = np.array(DirectionalIndicatorMinus(h.tolist(), l.tolist(), c.tolist(), 14))
    py_aroon_up = np.array(AroonUp(h.tolist(), 14))
    py_aroon_down = np.array(AroonDown(l.tolist(), 14))
    py_aroon_osc = np.array(AroonOsc(h.tolist(), l.tolist(), 14))
    py_parabolic = np.array(ParabolicSAR(h.tolist(), l.tolist(), 0.02, 0.02, 0.2))

    py_boll_up = np.array(BollingerUp(c.tolist(), 20, 2))
    py_boll_mid = np.array(BollingerMid(c.tolist(), 20))
    py_boll_down = np.array(BollingerDown(c.tolist(), 20, 2))
    py_boll_width = np.array(BollingerWidth(c.tolist(), 20, 2))

    py_mfi14 = np.array(MoneyFlowIndex(h.tolist(), l.tolist(), c.tolist(), v.tolist(), 14))
    py_obv = np.array(OBV(c.tolist(), v.tolist()))
    py_pvt = np.array(PVT(c.tolist(), v.tolist()))
    py_cmf20 = np.array(ChaikinMoneyFlow(h.tolist(), l.tolist(), c.tolist(), v.tolist(), 20))
    py_chaikin_osc = np.array(ChaikinOsc(h.tolist(), l.tolist(), c.tolist(), v.tolist(), 3, 10))
    py_emv14 = np.array(EaseOfMovement(h.tolist(), l.tolist(), v.tolist(), 14))
    py_ewo = np.array(ElliotWaveOscillator(h.tolist(), l.tolist(), 5, 35))
    py_env_up = np.array(EnvelopeUp(c.tolist(), 20, 2.5))
    py_env_mid = np.array(EnvelopeMid(c.tolist(), 20))
    py_env_down = np.array(EnvelopeDown(c.tolist(), 20, 2.5))
    py_kelt_up = np.array(KeltnerUp(h.tolist(), l.tolist(), c.tolist(), 20, 10, 2.0))
    py_kelt_down = np.array(KeltnerDown(h.tolist(), l.tolist(), c.tolist(), 20, 10, 2.0))
    py_klinger = np.array(KlingerOsc(h.tolist(), l.tolist(), c.tolist(), v.tolist(), 34, 55))
    py_linreg14 = np.array(LinearReg(c.tolist(), 14))
    py_linreg_slope14 = np.array(LinearRegSlope(c.tolist(), 14))
    py_massindex = np.array(MassIndex(h.tolist(), l.tolist(), 9, 25))
    py_pc_up = np.array(PriceChannelUp(h.tolist(), 20))
    py_pc_down = np.array(PriceChannelDown(l.tolist(), 20))
    py_qqef, py_qqes = QQEF(c.tolist(), 14, 5)
    py_qqef = np.array(py_qqef)
    py_qqes = np.array(py_qqes)
    py_trix15 = np.array(TRIX(c.tolist(), 15))
    py_chaikin_vol = np.array(ChaikinVolatility(h.tolist(), l.tolist(), 10, 10))

    # Slice by aligned indices
    def sl(arr):
        return arr[idx]

    tests = [
        ("SMA20", ideal_df["SMA20"].values[valid_mask], sl(py_sma20), 0.5),
        ("EMA20", ideal_df["EMA20"].values[valid_mask], sl(py_ema20), 0.5),
        ("DEMA20", ideal_df["DEMA20"].values[valid_mask], sl(py_dema20), 0.8),
        ("TEMA20", ideal_df["TEMA20"].values[valid_mask], sl(py_tema20), 0.8),
        ("HullMA20", ideal_df["HullMA20"].values[valid_mask], sl(py_hull20), 0.8),
        ("FRAMA", ideal_df["FRAMA"].values[valid_mask], sl(py_frama), 1.0),
        ("KAMA1030", ideal_df["KAMA1030"].values[valid_mask], sl(py_kama), 1.0),

        ("RSI14", ideal_df["RSI14"].values[valid_mask], sl(py_rsi14), 1.5),
        ("CCI20", ideal_df["CCI20"].values[valid_mask], np.round(sl(py_cci20), 2), 3.0, 0.12),
        ("Momentum10", ideal_df["Momentum10"].values[valid_mask], sl(py_mom10), 1.5),
        ("ROC10", ideal_df["ROC10"].values[valid_mask], sl(py_roc10), 2.0),
        ("CMO9", ideal_df["CMO9"].values[valid_mask], sl(py_cmo9), 3.0),
        ("WilliamsR14", ideal_df["WilliamsR14"].values[valid_mask], sl(py_willr14), 2.0),
        ("StochFast", ideal_df["StochFast"].values[valid_mask], sl(py_stoch_fast), 2.0),
        ("StochSlow", ideal_df["StochSlow"].values[valid_mask], sl(py_stoch_slow), 2.0),
        ("Qstick10", ideal_df["Qstick10"].values[valid_mask], sl(py_qstick10), 3.0),
        ("RVI10", ideal_df["RVI10"].values[valid_mask], sl(py_rvi10), 3.0),
        ("RVI10Sig", ideal_df["RVI10Sig"].values[valid_mask], sl(py_rvi10_sig), 3.0),
        ("MACD1226", ideal_df["MACD1226"].values[valid_mask], np.round(sl(py_macd1226), 2), 3.0, 0.02),

        ("HHV20", ideal_df["HHV20"].values[valid_mask], sl(py_hhv20), 0.5),
        ("LLV20", ideal_df["LLV20"].values[valid_mask], sl(py_llv20), 0.5),
        ("ATR14", ideal_df["ATR14"].values[valid_mask], sl(py_atr14), 1.0),
        ("ADX14", ideal_df["ADX14"].values[valid_mask], sl(py_adx14), 2.0),
        ("DIPlus14", ideal_df["DIPlus14"].values[valid_mask], sl(py_di_plus), 2.0),
        ("DIMinus14", ideal_df["DIMinus14"].values[valid_mask], sl(py_di_minus), 2.0),
        ("AroonUp14", ideal_df["AroonUp14"].values[valid_mask], sl(py_aroon_up), 2.0),
        ("AroonDown14", ideal_df["AroonDown14"].values[valid_mask], sl(py_aroon_down), 2.0),
        ("AroonOsc14", ideal_df["AroonOsc14"].values[valid_mask], sl(py_aroon_osc), 3.0),
        ("Parabolic", ideal_df["Parabolic"].values[valid_mask], sl(py_parabolic), 1.0),

        ("BollUp", ideal_df["BollUp"].values[valid_mask], sl(py_boll_up), 0.8),
        ("BollMid", ideal_df["BollMid"].values[valid_mask], sl(py_boll_mid), 0.8),
        ("BollDown", ideal_df["BollDown"].values[valid_mask], sl(py_boll_down), 0.8),
        ("BollWidth", ideal_df["BollWidth"].values[valid_mask], sl(py_boll_width), 3.0),

        ("MFI14", ideal_df["MFI14"].values[valid_mask], sl(py_mfi14), 2.0),
        ("OBV", ideal_df["OBV"].values[valid_mask], sl(py_obv), 1.0),
        ("PVT", ideal_df["PVT"].values[valid_mask], sl(py_pvt), 1.0),
        ("ChaikinMF20", ideal_df["ChaikinMF20"].values[valid_mask], sl(py_cmf20), 3.0),
        ("ChaikinOsc", ideal_df["ChaikinOsc"].values[valid_mask], sl(py_chaikin_osc), 3.0),
        ("EaseOfMovement14", ideal_df["EaseOfMovement14"].values[valid_mask], sl(py_emv14), 5.0),
        ("ElliotWaveOsc_5_35", ideal_df["ElliotWaveOsc_5_35"].values[valid_mask], sl(py_ewo), 5.0),
        ("EnvelopeUp20_2p5", ideal_df["EnvelopeUp20_2p5"].values[valid_mask], sl(py_env_up), 3.0),
        ("EnvelopeMid20_2p5", ideal_df["EnvelopeMid20_2p5"].values[valid_mask], sl(py_env_mid), 3.0),
        ("EnvelopeDown20_2p5", ideal_df["EnvelopeDown20_2p5"].values[valid_mask], sl(py_env_down), 3.0),
        ("KeltnerUp20", ideal_df["KeltnerUp20"].values[valid_mask], sl(py_kelt_up), 5.0),
        ("KeltnerDown20", ideal_df["KeltnerDown20"].values[valid_mask], sl(py_kelt_down), 5.0),
        ("KlingerOsc34", ideal_df["KlingerOsc34"].values[valid_mask], sl(py_klinger), 5.0),
        ("LinearReg14", ideal_df["LinearReg14"].values[valid_mask], sl(py_linreg14), 3.0),
        ("LinearRegSlope14", ideal_df["LinearRegSlope14"].values[valid_mask], sl(py_linreg_slope14), 3.0),
        ("MassIndex9", ideal_df["MassIndex9"].values[valid_mask], sl(py_massindex), 5.0),
        ("PriceChannelUp20", ideal_df["PriceChannelUp20"].values[valid_mask], sl(py_pc_up), 3.0),
        ("PriceChannelDown20", ideal_df["PriceChannelDown20"].values[valid_mask], sl(py_pc_down), 3.0),
        ("QQEF14_5", ideal_df["QQEF14_5"].values[valid_mask], sl(py_qqef), 5.0),
        ("QQES14_5", ideal_df["QQES14_5"].values[valid_mask], sl(py_qqes), 5.0),
        ("TRIX15", ideal_df["TRIX15"].values[valid_mask], sl(py_trix15), 5.0),
        ("ChaikinVol10_10", ideal_df["ChaikinVol10_10"].values[valid_mask], sl(py_chaikin_vol), 5.0),
    ]

    print("\n=== IDEALDATA EXPORT KALİBRASYON RAPORU ===")
    print(f"Eşleşen bar sayısı: {valid_mask.sum()}/{len(ideal_df)}")
    print("-" * 100)
    print(f"{'İndikatör':14} | {'Durum':5} | {'Max %':8} | {'Ort %':8} | {'MaxAbs':10} | {'Örnek':6}")
    print("-" * 100)

    results = []
    for item in tests:
        if len(item) == 4:
            name, ideal_vals, py_vals, tol_pct = item
            tol_abs = 0.01
        else:
            name, ideal_vals, py_vals, tol_pct, tol_abs = item

        res = compare_indicator(name, ideal_vals, py_vals, tol_pct=tol_pct, tol_abs=tol_abs)
        results.append(res)
        icon = "OK" if res["is_match"] else "XX"
        print(f"[{icon}] {name:12} | {res['status']:5} | {res['max_pct_diff']:8.3f} | {res['mean_pct_diff']:8.3f} | {res['max_abs_diff']:10.4f} | {res['sample_count']:6}")

    # PVT01_14 not implemented in Python yet
    if "PVT01_14" in ideal_df.columns:
        print("\n[NOT] PVT01_14 için Python karşılığı henüz yok (skip).")

    passed = sum(1 for r in results if r["is_match"])
    total = len(results)
    print("-" * 100)
    print(f"SONUÇ: {passed}/{total} başarılı")


if __name__ == "__main__":
    run()
