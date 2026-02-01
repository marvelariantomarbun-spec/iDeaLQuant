// =============================================================================
// IdealData Indicator Export (PART2)
// =============================================================================
// Paste into IdealData as a new system and run on 1DK chart.
// Output: D:\Projects\IdealQuant\data\ideal_ind_export_part2.csv
// =============================================================================

var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var Vol = Sistem.GrafikFiyatSec("Hacim");
var Lot = Sistem.GrafikFiyatSec("Lot");

// --- INDICATORS (PART2) ---
var ADX14 = Sistem.ADX(14);
var AroonUp25 = Sistem.AroonUp(25);
var AroonDown25 = Sistem.AroonDown(25);
var AroonOsc25 = Sistem.AroonOsc(25);

var BollingerUp20 = Sistem.BollingerUp("Simple", 20, 2);
var BollingerMid20 = Sistem.BollingerMid("Simple", 20, 2);
var BollingerDown20 = Sistem.BollingerDown("Simple", 20, 2);
var BollingerWidth20 = Sistem.BollingerWidth(20, 2);

var ChaikinMoneyFlow20 = Sistem.ChaikinMoneyFlow(20);
var ChaikinVolatility10_10 = Sistem.ChaikinVolatility(10, 10);
var ChandeMomentum9 = Sistem.ChandeMomentum(9);

var DEMA20 = Sistem.DEMA(20);
var DIPlus14 = Sistem.DirectionalIndicatorPlus(14);
var DIMinus14 = Sistem.DirectionalIndicatorMinus(14);

var EaseOfMovement14 = Sistem.EaseOfMovement(14);
var ElliotWaveOsc_5_35 = Sistem.ElliotWaveOscillator(5, 35);

var EnvelopeUp20_2p5 = Sistem.EnvelopeUp("Simple", 20, 2.5);
var EnvelopeMid20_2p5 = Sistem.EnvelopeMid("Simple", 20, 2.5);
var EnvelopeDown20_2p5 = Sistem.EnvelopeDown("Simple", 20, 2.5);

var HHV20 = Sistem.HHV(20);
var HullMA20 = Sistem.HullMA(20);
var KAMA1030 = Sistem.KAMA(10, 2, 30);

var KeltnerUp20 = Sistem.KeltnerUp(20);
var KeltnerDown20 = Sistem.KeltnerDown(20);
var KlingerOsc34 = Sistem.KlingerOsc(34);

var LLV20 = Sistem.LLV(20);
var LinearReg14 = Sistem.LinearReg(14);
var LinearRegSlope14 = Sistem.LinearRegSlope(14);

var MA20 = Sistem.MA(C, "Simple", 20);
var MACD1226 = Sistem.MACD(12, 26);
var MassIndex9 = Sistem.MassIndex(9);
var Momentum10 = Sistem.Momentum(10);
var MoneyFlowIndex14 = Sistem.MoneyFlowIndex(14);

var PriceChannelUp20 = Sistem.PriceChannelUp(20);
var PriceChannelDown20 = Sistem.PriceChannelDown(20);

var QQEF14_5 = Sistem.QQEF(14, 5);
var QQES14_5 = Sistem.QQES(14, 5);

var Qstick10 = Sistem.Qstick(10);
var RSI14 = Sistem.RSI(14);
var StochFast14 = Sistem.StochasticFast(14, 3);
var StochSlow14 = Sistem.StochasticSlow(14, 3);

var TEMA20 = Sistem.TEMA(20);
var TRIX15 = Sistem.TRIX(15);
var WilliamsR14 = Sistem.WilliamsR(14);

// --- WRITE CSV ---
var sb = new System.Text.StringBuilder();
sb.AppendLine("BarNo;Date;Time;Close;Hacim;Lot;ADX;AroonUp;AroonDown;AroonOsc;BollingerUp;BollingerMid;BollingerDown;BollingerWidth;ChaikinMoneyFlow;ChaikinVolatility;ChandeMomentum;DEMA;DirectionalIndicatorPlus;DirectionalIndicatorMinus;EaseOfMovement;ElliotWaveOscillator;EnvelopeUp;EnvelopeMid;EnvelopeDown;HHV;HullMA;KAMA;KeltnerUp;KeltnerDown;KlingerOsc;LLV;LinearReg;LinearRegSlope;MA;MACD;MassIndex;Momentum;MoneyFlowIndex;PriceChannelUp;PriceChannelDown;QQEF;QQES;Qstick;RSI;StochasticFast;StochasticSlow;TEMA;TRIX;WilliamsR");

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

	sb.Append(ADX14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(AroonUp25[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(AroonDown25[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(AroonOsc25[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(BollingerUp20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(BollingerMid20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(BollingerDown20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(BollingerWidth20[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(ChaikinMoneyFlow20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(ChaikinVolatility10_10[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(ChandeMomentum9[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(DEMA20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(DIPlus14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(DIMinus14[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(EaseOfMovement14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(ElliotWaveOsc_5_35[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(EnvelopeUp20_2p5[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(EnvelopeMid20_2p5[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(EnvelopeDown20_2p5[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(HHV20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(HullMA20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(KAMA1030[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(KeltnerUp20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(KeltnerDown20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(KlingerOsc34[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(LLV20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(LinearReg14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(LinearRegSlope14[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(MA20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(MACD1226[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(MassIndex9[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(Momentum10[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(MoneyFlowIndex14[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(PriceChannelUp20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(PriceChannelDown20[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(QQEF14_5[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(QQES14_5[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(Qstick10[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(RSI14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(StochFast14[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(StochSlow14[i].ToString("0.00").Replace(",", ".") + ";");

	sb.Append(TEMA20[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(TRIX15[i].ToString("0.00").Replace(",", ".") + ";");
	sb.Append(WilliamsR14[i].ToString("0.00").Replace(",", "."));
	sb.AppendLine();
}

string path = @"D:\Projects\IdealQuant\data\ideal_ind_export_part2.csv";
try
{
	System.IO.File.WriteAllText(path, sb.ToString());
	Sistem.Mesaj("OK: " + path);
}
catch (Exception ex)
{
	Sistem.Mesaj("Hata: " + ex.Message);
}
