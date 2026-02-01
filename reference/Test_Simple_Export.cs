// ============================================================
// BASIT TEST - Önce bu çalışıyor mu kontrol edelim
// ============================================================

var C = Sistem.GrafikFiyatSec("Kapanis");
var ema20 = Sistem.MA("Exp", 20);
var rsi14 = Sistem.RSI(14);

var sb = new System.Text.StringBuilder();
sb.AppendLine("BarNo;Close;EMA20;RSI14");

int start = Math.Max(50, Sistem.BarSayisi - 100);

for (int i = start; i < Sistem.BarSayisi; i++)
{
    sb.Append(i.ToString() + ";");
    sb.Append(C[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(ema20[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(rsi14[i].ToString("0.00").Replace(",", "."));
    sb.AppendLine();
}

// Önce C:\Temp klasörüne yazmayı dene (izin sorunu olabilir)
string path = @"C:\Temp\test_export.csv";
try {
    System.IO.File.WriteAllText(path, sb.ToString());
    Sistem.Mesaj("BASARILI: " + path);
} catch (Exception ex) {
    Sistem.Mesaj("HATA: " + ex.Message);
}
