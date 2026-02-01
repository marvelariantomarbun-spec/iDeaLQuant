// =============================================================================
// MFI Volume Test - Hacim mi Lot mu?
// =============================================================================
// IdealData'da MFI hangi volume verisini kullanıyor test et
// Paste into IdealData as a new system and run on 1DK chart.
// Output: D:\Projects\IdealQuant\data\mfi_volume_test.csv
// =============================================================================

var V = Sistem.GrafikVerileri;
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var Vol = Sistem.GrafikFiyatSec("Hacim");
var Lot = Sistem.GrafikFiyatSec("Lot");

// MFI ile otomatik volume
var MFI_Auto = Sistem.MoneyFlowIndex(14);

// Typical Price hesapla (MFI için)
var TypicalPrice = Sistem.Liste(0);
for (int i = 0; i < Sistem.BarSayisi; i++)
{
    TypicalPrice[i] = (H[i] + L[i] + C[i]) / 3;
}

// Raw Money Flow = Typical Price * Volume
var RawMF_Hacim = Sistem.Liste(0);
var RawMF_Lot = Sistem.Liste(0);
for (int i = 0; i < Sistem.BarSayisi; i++)
{
    RawMF_Hacim[i] = TypicalPrice[i] * Vol[i];
    RawMF_Lot[i] = TypicalPrice[i] * Lot[i];
}

// --- WRITE CSV ---
var sb = new System.Text.StringBuilder();
sb.AppendLine("BarNo;Date;Time;Close;Hacim;Lot;TypicalPrice;RawMF_Hacim;RawMF_Lot;MFI_Auto");

int start = Math.Max(50, Sistem.BarSayisi - 1000);

for (int i = start; i < Sistem.BarSayisi; i++)
{
    string date = V[i].Date.ToString("dd.MM.yyyy");
    string time = V[i].Date.ToString("HH:mm");

    sb.Append(i + ";");
    sb.Append(date + ";" + time + ";");
    sb.Append(C[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Vol[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(Lot[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(TypicalPrice[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(RawMF_Hacim[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(RawMF_Lot[i].ToString("0.00").Replace(",", ".") + ";");
    sb.Append(MFI_Auto[i].ToString("0.00").Replace(",", "."));
    sb.AppendLine();
}

string path = @"D:\Projects\IdealQuant\data\mfi_volume_test.csv";
try
{
    System.IO.File.WriteAllText(path, sb.ToString());
    Sistem.Mesaj("OK: " + path);
}
catch (Exception ex)
{
    Sistem.Mesaj("Hata: " + ex.Message);
}
