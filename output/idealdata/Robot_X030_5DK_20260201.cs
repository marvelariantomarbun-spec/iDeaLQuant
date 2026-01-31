// ===============================================================================================
// ROBOT: S1 + S2 BİRLEŞİK İŞLEM ROBOTU
// ===============================================================================================
// Strateji 1: S1_X030_5DK_ENDEKS_20260201 (Gatekeeper - Yön Belirler)
// Strateji 2: S2_X030_5DK_ENDEKS_20260201 (İşlem Motoru)
// Sembol: VIP'VIP-X030
// Periyot: 5 dakika
// Lot: 1
// Oluşturma: 2026-02-01 02:00
// ===============================================================================================

// --- AYARLAR ---
var LotSize = 1;
var Sembol = "VIP'VIP-X030";
var Periyot = "5";

// --- STRATEJİLERİ GETİR ---
var Sistem1 = Sistem.SistemGetir("S1_X030_5DK_ENDEKS_20260201", Sembol, Periyot);
var Sistem2 = Sistem.SistemGetir("S2_X030_5DK_ENDEKS_20260201", Sembol, Periyot);

if (Sistem1 == null || Sistem2 == null)
{
    Sistem.Mesaj(Sistem.Name + " - Strateji bulunamadı!", Color.Red);
    return null;
}

// --- SAAT KONTROLÜ ---
var Saat = DateTime.Now.TimeOfDay;
bool SeansSaati = (Saat >= new TimeSpan(9, 30, 0) && Saat < new TimeSpan(18, 15, 0)) ||
                  (Saat >= new TimeSpan(19, 0, 0) && Saat < new TimeSpan(23, 0, 0));

if (!SeansSaati)
{
    return null;
}

// --- SİNYAL BİRLEŞTİRME ---
// S1: Kapı açık mı? (Yön belirleme)
// S2: İşlem sinyali var mı?

var Yon1 = Sistem1.SonYon;  // A, S veya F
var Yon2 = Sistem2.SonYon;  // A, S veya F

string Sinyal = "";

// Kapı LONG açık ve S2 LONG sinyali
if (Yon1 == "A" && Yon2 == "A")
{
    Sinyal = "A";
}
// Kapı SHORT açık ve S2 SHORT sinyali
else if (Yon1 == "S" && Yon2 == "S")
{
    Sinyal = "S";
}
// Kapı kapalı veya sinyal uyuşmazlığı
else
{
    Sinyal = "F";
}

// --- POZİSYON YÖNETİMİ ---
var EmirSembol = Sembol;
var SonFiyat = Sistem.SonFiyat(EmirSembol);
var Anahtar = Sistem.Name + "," + EmirSembol;

double IslemFiyat = 0;
DateTime IslemTarih;
var Rezerv = "";
var Pozisyon = Sistem.PozisyonKontrolOku(Anahtar, out IslemFiyat, out IslemTarih);

double Miktar = 0;

if (Sinyal == "F" && Pozisyon != 0)
{
    // Pozisyonu kapat
    Miktar = -Pozisyon;
    Rezerv = "POZİSYON KAPATILDI";
}
else if (Sinyal == "A" && Pozisyon != LotSize)
{
    // Long aç/artır
    Miktar = LotSize - Pozisyon;
    Rezerv = "LONG AÇ";
}
else if (Sinyal == "S" && Pozisyon != -LotSize)
{
    // Short aç/artır
    Miktar = -LotSize - Pozisyon;
    Rezerv = "SHORT AÇ";
}

// --- EMİR GÖNDER ---
if (Miktar != 0)
{
    var Islem = Miktar > 0 ? "ALIS" : "SATIS";
    
    Sistem.PozisyonKontrolGuncelle(Anahtar, Miktar + Pozisyon, SonFiyat, Rezerv);
    
    Sistem.EmirSembol = EmirSembol;
    Sistem.EmirIslem = Islem;
    Sistem.EmirSuresi = "KIE";
    Sistem.EmirTipi = "Piyasa";
    Sistem.EmirMiktari = Math.Abs(Miktar);
    Sistem.EmirGonder();
    
    Sistem.Mesaj(Sistem.Name + " | " + Rezerv + " | Lot: " + Math.Abs(Miktar), Color.Green);
}

// --- BİLGİ PANELİ ---
string panelInfo = "══════ S1+S2 ROBOT ══════" + Environment.NewLine +
                   "S1 (Kapı): " + Yon1 + Environment.NewLine +
                   "S2 (İşlem): " + Yon2 + Environment.NewLine +
                   "─────────────────────" + Environment.NewLine +
                   "Sinyal: " + Sinyal + Environment.NewLine +
                   "Pozisyon: " + Pozisyon + Environment.NewLine +
                   "Fiyat: " + SonFiyat.ToString("0.00");

var panelRenk = Color.DarkBlue;
if (Pozisyon > 0) panelRenk = Color.DarkGreen;
else if (Pozisyon < 0) panelRenk = Color.DarkRed;

Sistem.Dortgen(1, 10, 30, 180, 120, panelRenk, Color.Black, Color.White);
Sistem.GradientYaziEkle(panelInfo, 1, 15, 35, Color.White, Color.LightBlue, "Consolas", 9);

return null;
