
// =============================================================================
// PART 1 - Sadece Temel Göstergeler (Test)
// =============================================================================

var V = Sistem.GrafikVerileri;
var C = Sistem.GrafikFiyatSec("Kapanis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");

// Sadece çalıştığını bildiğimiz göstergeler
var sma20 = Sistem.MA("Simple", 20);
var ema20 = Sistem.MA("Exp", 20);
var rsi14 = Sistem.RSI(14);
var momentum = Sistem.Momentum(10);
var hhv20 = Sistem.HHV(20);
var llv20 = Sistem.LLV(20);
var atr14 = Sistem.AverageTrueRange(14);
var adx14 = Sistem.ADX(14);

var sb = new System.Text.StringBuilder();
sb.AppendLine("Date;Time;Close;SMA20;EMA20;RSI14;Momentum;HHV20;LLV20;ATR14;ADX14");

int start = Math.Max(50, Sistem.BarSayisi - 1000);

for (int i = start; i < Sistem.BarSayisi; i++)
{
    string date = V[i].Date.ToString("dd.MM.yyyy");
    string time = V[i].Date.ToString("HH:mm");
    
    sb.Append(date + ";" + time + ";");
    sb.Append(C[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(sma20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ema20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(rsi14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(momentum[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(hhv20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(llv20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(atr14[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(adx14[i].ToString("0.00").Replace(",", "."));
    sb.AppendLine();
}

string path = @"D:\Projects\IdealQuant\data\ideal_indicators_part1.csv";
try {
    System.IO.File.WriteAllText(path, sb.ToString());
    Sistem.Mesaj("OK: " + path);
} catch (Exception ex) {
    Sistem.Mesaj("Hata: " + ex.Message);
}
