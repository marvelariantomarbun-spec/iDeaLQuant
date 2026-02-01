
// =============================================================================
// IdealData Indicator Export (SAFE / ARS_v2 format)
// =============================================================================
// Paste into IdealData as a new system and run on 1DK chart.
// Output: D:\Projects\IdealQuant\data\ideal_ind_export.csv
// =============================================================================

var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var T = Sistem.GrafikFiyatSec("Tipik");
var Vol = Sistem.GrafikFiyatSec("Hacim");
var Lot = Sistem.GrafikFiyatSec("Lot");

// --- INDICATORS (verified in CxSistem.txt) ---
var SMA20 = Sistem.MA(C, "Simple", 20);
var EMA20 = Sistem.MA(C, "Exp", 20);
var DEMA20 = Sistem.DEMA(20);
var TEMA20 = Sistem.TEMA(20);
var HullMA20 = Sistem.HullMA(20);
var FRAMA = Sistem.FRAMA();
var KAMA1030 = Sistem.KAMA(10, 2, 30);

var RSI14 = Sistem.RSI(14);
var CCI20 = Sistem.CommodityChannelIndex(20);
var Momentum10 = Sistem.Momentum(10);
var ROC10 = Sistem.PriceRocPercent(10);
var CMO9 = Sistem.ChandeMomentum(9);
var WilliamsR14 = Sistem.WilliamsR(14);
var StochFast = Sistem.StochasticFast(14, 3);
var StochSlow = Sistem.StochasticSlow(14, 3);
var Qstick10 = Sistem.Qstick(10);
var RVI10 = Sistem.RelativeVigorIndex(10);
var RVI10Sig = Sistem.RelativeVigorIndexSignal(10);

var MACD1226 = Sistem.MACD(12, 26);

var HHV20 = Sistem.HHV(20);
var LLV20 = Sistem.LLV(20);
var ATR14 = Sistem.AverageTrueRange(14);
var ADX14 = Sistem.ADX(14);
var DIPlus14 = Sistem.DirectionalIndicatorPlus(14);
var DIMinus14 = Sistem.DirectionalIndicatorMinus(14);
var AroonUp14 = Sistem.AroonUp(14);
var AroonDown14 = Sistem.AroonDown(14);
var AroonOsc14 = Sistem.AroonOsc(14);
var Parabolic = Sistem.Parabolic(0.02, 0.2);

var BollUp = Sistem.BollingerUp("Simple", 20, 2);
var BollMid = Sistem.BollingerMid("Simple", 20, 2);
var BollDown = Sistem.BollingerDown("Simple", 20, 2);
var BollWidth = Sistem.BollingerWidth(20, 2);

var MFI14 = Sistem.MoneyFlowIndex(14);
var OBV = Sistem.OnBalanceVolume();
var PVT = Sistem.PriceVolumeTrend();
var PVT01_14 = Sistem.PVT01(14);
var ChaikinMF20 = Sistem.ChaikinMoneyFlow(20);
var ChaikinOsc = Sistem.ChaikinOsc();
var ADL = Sistem.AccumulationDistribution();
var EaseOfMovement14 = Sistem.EaseOfMovement(14);
var ElliotWaveOsc_5_35 = Sistem.ElliotWaveOscillator(5, 35);
var EnvelopeUp20_2p5 = Sistem.EnvelopeUp("Simple", 20, 2.5);
var EnvelopeMid20_2p5 = Sistem.EnvelopeMid("Simple", 20, 2.5);
var EnvelopeDown20_2p5 = Sistem.EnvelopeDown("Simple", 20, 2.5);
var KeltnerUp20 = Sistem.KeltnerUp(20);
var KeltnerDown20 = Sistem.KeltnerDown(20);
var KlingerOsc34 = Sistem.KlingerOsc(34);
var LinearReg14 = Sistem.LinearReg(14);
var LinearRegSlope14 = Sistem.LinearRegSlope(14);
var MassIndex9 = Sistem.MassIndex(9);
var PriceChannelUp20 = Sistem.PriceChannelUp(20);
var PriceChannelDown20 = Sistem.PriceChannelDown(20);
var QQEF14_5 = Sistem.QQEF(14, 5);
var QQES14_5 = Sistem.QQES(14, 5);
var TRIX15 = Sistem.TRIX(15);
var ChaikinVol10_10 = Sistem.ChaikinVolatility(10, 10);

// --- WRITE CSV ---
var sb = new System.Text.StringBuilder();
sb.AppendLine("BarNo;Date;Time;Close;Hacim;Lot;SMA20;EMA20;DEMA20;TEMA20;HullMA20;FRAMA;KAMA1030;RSI14;CCI20;Momentum10;ROC10;CMO9;WilliamsR14;StochFast;StochSlow;Qstick10;RVI10;RVI10Sig;MACD1226;HHV20;LLV20;ATR14;ADX14;DIPlus14;DIMinus14;AroonUp14;AroonDown14;AroonOsc14;Parabolic;BollUp;BollMid;BollDown;BollWidth;MFI14;OBV;PVT;PVT01_14;ChaikinMF20;ChaikinOsc;ADL;EaseOfMovement14;ElliotWaveOsc_5_35;EnvelopeUp20_2p5;EnvelopeMid20_2p5;EnvelopeDown20_2p5;KeltnerUp20;KeltnerDown20;KlingerOsc34;LinearReg14;LinearRegSlope14;MassIndex9;PriceChannelUp20;PriceChannelDown20;QQEF14_5;QQES14_5;TRIX15;ChaikinVol10_10");

int start = Math.Max(50, Sistem.BarSayisi - 5000);

for (int i = start; i < Sistem.BarSayisi; i++)
{
    string date = V[i].Date.ToString("dd.MM.yyyy");
    string time = V[i].Date.ToString("HH:mm");

    sb.Append(i + ";");
    sb.Append(date + ";" + time + ";");
    sb.Append(C[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Vol[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Lot[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(SMA20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(EMA20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(DEMA20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(TEMA20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(HullMA20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(FRAMA[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(KAMA1030[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(RSI14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(CCI20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Momentum10[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ROC10[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(CMO9[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(WilliamsR14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(StochFast[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(StochSlow[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Qstick10[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(RVI10[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(RVI10Sig[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(MACD1226[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(HHV20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(LLV20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ATR14[i].ToString("0.000000").Replace(",", ".") + ";");
    sb.Append(ADX14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(DIPlus14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(DIMinus14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(AroonUp14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(AroonDown14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(AroonOsc14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Parabolic[i].ToString("0.000000").Replace(",", ".") + ";");
    sb.Append(BollUp[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(BollMid[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(BollDown[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(BollWidth[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(MFI14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(OBV[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(PVT[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(PVT01_14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ChaikinMF20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ChaikinOsc[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ADL[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(EaseOfMovement14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ElliotWaveOsc_5_35[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(EnvelopeUp20_2p5[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(EnvelopeMid20_2p5[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(EnvelopeDown20_2p5[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(KeltnerUp20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(KeltnerDown20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(KlingerOsc34[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(LinearReg14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(LinearRegSlope14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(MassIndex9[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(PriceChannelUp20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(PriceChannelDown20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(QQEF14_5[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(QQES14_5[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(TRIX15[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ChaikinVol10_10[i].ToString("0.00").Replace(",", "."));
    sb.AppendLine();
}

string path = @"D:\Projects\IdealQuant\data\ideal_ind_export.csv";
try
{
    System.IO.File.WriteAllText(path, sb.ToString());
    Sistem.Mesaj("OK: " + path);
}
catch (Exception ex)
{
    Sistem.Mesaj("Hata: " + ex.Message);
}
