// ===============================================================================================
// STRATEJİ 2: ARS TREND TAKİP SİSTEMİ v4.1
// ===============================================================================================
// Sembol: VIP_X030-T
// Periyot: 1 dakika
// Vade Tipi: ENDEKS
// Oluşturma: 2026-02-13 23:09
// ===============================================================================================

// --- VADE TİPİ ---
string VadeTipi = "ENDEKS";

// --- ATR EXIT PARAMETRELER ---
int ATR_Exit_Period = 100;
double ATR_SL_Mult = 3.5;
double ATR_TP_Mult = 5.0;
double ATR_Trail_Mult = 3.5;
int Exit_Confirm_Bars = 1;
double Exit_Confirm_Mult = 0.5;

// --- ARS PARAMETRELER ---
int ARS_EMA_Period = 60;
int ARS_ATR_Period = 45;
double ARS_ATR_Mult = 0.3;
double ARS_Min_Band = 0.003;
double ARS_Max_Band = 0.01;

// --- GİRİŞ SİNYALİ PARAMETRELER ---
int MOMENTUM_Period = 20;
double MOMENTUM_THRESHOLD = 100.0;
double MOMENTUM_BASE = 100.0;
int BREAKOUT_Period = 50;
double VOLUME_MULT = 0.8;

// ===============================================================================================
// DİNAMİK BAYRAM TARİHLERİ (2024-2030)
// ===============================================================================================
int yil = DateTime.Now.Year;
DateTime Ramazan, Kurban;

switch(yil)
{
    case 2024: Ramazan = new DateTime(2024, 4, 10); Kurban = new DateTime(2024, 6, 16); break;
    case 2025: Ramazan = new DateTime(2025, 3, 30); Kurban = new DateTime(2025, 6, 6); break;
    case 2026: Ramazan = new DateTime(2026, 3, 20); Kurban = new DateTime(2026, 5, 27); break;
    case 2027: Ramazan = new DateTime(2027, 3, 9); Kurban = new DateTime(2027, 5, 16); break;
    case 2028: Ramazan = new DateTime(2028, 2, 26); Kurban = new DateTime(2028, 5, 5); break;
    case 2029: Ramazan = new DateTime(2029, 2, 14); Kurban = new DateTime(2029, 4, 24); break;
    case 2030: Ramazan = new DateTime(2030, 2, 3); Kurban = new DateTime(2030, 4, 13); break;
    default: Ramazan = new DateTime(yil, 3, 15); Kurban = new DateTime(yil, 5, 20); break;
}

DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] { "01.01","04.23","05.01","05.19","07.15","08.30","10.29" };

// ===============================================================================================
// VERİ HAZIRLIĞI
// ===============================================================================================
var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var T = Sistem.GrafikFiyatSec("Tipik");
var Lot = Sistem.GrafikFiyatSec("Lot");

// ===============================================================================================
// ARS HESAPLAMA
// ===============================================================================================
var ATR = Sistem.AverageTrueRange(ARS_ATR_Period);
var ARS_EMA = Sistem.MA(T, "Exp", ARS_EMA_Period);
var ARS = Sistem.Liste(0);

for (int i = 1; i < Sistem.BarSayisi; i++)
{
    float dinamikK;
    if (ARS_ATR_Mult > 0) {
        dinamikK = (ATR[i] / ARS_EMA[i]) * (float)ARS_ATR_Mult;
        dinamikK = Math.Max((float)ARS_Min_Band, Math.Min((float)ARS_Max_Band, dinamikK));
    } else {
        dinamikK = (float)ARS_Min_Band;
    }
    
    float altBand = ARS_EMA[i] * (1 - dinamikK);
    float ustBand = ARS_EMA[i] * (1 + dinamikK);
    
    if (altBand > ARS[i - 1])
        ARS[i] = altBand;
    else if (ustBand < ARS[i - 1])
        ARS[i] = ustBand;
    else
        ARS[i] = ARS[i - 1];
    
    float roundStep = ARS_ATR_Mult > 0 ? Math.Max(0.01f, ATR[i] * 0.1f) : 0.025f;
    ARS[i] = Sistem.SayiYuvarla(ARS[i], roundStep);
}

// TREND BELİRLEME
var TrendYonu = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++)
{
    if (C[i] > ARS[i]) TrendYonu[i] = 1;
    else if (C[i] < ARS[i]) TrendYonu[i] = -1;
    else TrendYonu[i] = TrendYonu[i-1];
}

var ATR_Exit = Sistem.AverageTrueRange(ATR_Exit_Period);

// GİRİŞ SİNYAL İNDİKATÖRLERİ
var Momentum = Sistem.Momentum(MOMENTUM_Period);
var HHV = Sistem.HHV(BREAKOUT_Period);
var LLV = Sistem.LLV(BREAKOUT_Period);

var MFI = Sistem.MoneyFlowIndex(90);
var MFI_HHV = Sistem.HHV(50, MFI);
var MFI_LLV = Sistem.LLV(50, MFI);

var Vol_HHV = Sistem.HHV(14, Lot);

// ===============================================================================================
// VADE SONU İŞ GÜNÜ HESAPLAMA
// ===============================================================================================
Func<DateTime, DateTime> VadeSonuIsGunu = (dt) =>
{
    var aySonu = new DateTime(dt.Year, dt.Month, DateTime.DaysInMonth(dt.Year, dt.Month));
    var d = aySonu;
    
    for (int k = 0; k < 15; k++)
    {
        if (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
        { d = d.AddDays(-1); continue; }
        
        string mmdd = d.ToString("MM.dd");
        bool tatil = false;
        for (int t = 0; t < resmiTatiller.Length; t++)
            if (resmiTatiller[t] == mmdd) { tatil = true; break; }
        if (tatil) { d = d.AddDays(-1); continue; }
        
        if ((d >= R2024 && d <= R2024.AddDays(3)) || (d >= K2024 && d <= K2024.AddDays(4)) ||
            (d >= R2025 && d <= R2025.AddDays(3)) || (d >= K2025 && d <= K2025.AddDays(4)) ||
            (d >= R2026 && d <= R2026.AddDays(3)) || (d >= K2026 && d <= K2026.AddDays(4)) ||
            (d >= R2027 && d <= R2027.AddDays(3)) || (d >= K2027 && d <= K2027.AddDays(4)))
        { d = d.AddDays(-1); continue; }
        
        break;
    }
    return d.Date;
};

// ===============================================================================================
// SİNYAL ÜRETİM DÖNGÜSÜ
// ===============================================================================================
for (int i = 1; i < V.Count; i++) Sistem.Yon[i] = "";

var Sinyal = "";
var SonYon = "";

float entryPrice = 0;
int entryBar = 0;
float extremePrice = 0;
int belowArsCount = 0;
int aboveArsCount = 0;

int vadeCooldownBar = Math.Max(ARS_EMA_Period, Math.Max(BREAKOUT_Period, Math.Max(ARS_ATR_Period, MOMENTUM_Period))) + 10;
int warmupBars = vadeCooldownBar;
int warmupBaslangicBar = -999;
bool warmupAktif = false;
bool arefeFlat = false;

for (int i = warmupBars; i < V.Count; i++)
{
    Sinyal = "";
    var dt = V[i].Date;
    var t = dt.TimeOfDay;
    
    bool gunSeansi = t >= new TimeSpan(9,30,0) && t < new TimeSpan(18,15,0);
    bool aksamSeansi = t >= new TimeSpan(19,0,0) && t < new TimeSpan(23,0,0);
    if (!(gunSeansi || aksamSeansi)) continue;
    
    bool vadeAyi = (VadeTipi == "SPOT") || (dt.Month % 2 == 0);
    bool vadeSonuGun = vadeAyi && (dt.Date == VadeSonuIsGunu(dt));
    
    bool arefe = dt.Date == R2024.AddDays(-1).Date || dt.Date == K2024.AddDays(-1).Date ||
                 dt.Date == R2025.AddDays(-1).Date || dt.Date == K2025.AddDays(-1).Date ||
                 dt.Date == R2026.AddDays(-1).Date || dt.Date == K2026.AddDays(-1).Date ||
                 dt.Date == R2027.AddDays(-1).Date || dt.Date == K2027.AddDays(-1).Date;
    
    if (arefe && vadeSonuGun && t > new TimeSpan(11,30,0))
    {
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }
    
    if (Sinyal == "F")
    {
        if (SonYon != Sinyal) { Sistem.Yon[i] = Sinyal; SonYon = Sinyal; }
        continue;
    }
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0)))
        continue;
    
    if (warmupAktif && warmupBaslangicBar == -999)
    {
        bool yeniSeansBaslangici = false;
        if (aksamSeansi && i > 0 && V[i-1].Date.TimeOfDay < new TimeSpan(19,0,0))
            yeniSeansBaslangici = true;
        if (gunSeansi && t >= new TimeSpan(9,30,0) && t < new TimeSpan(9,35,0))
            if (i > 0 && dt.Date != V[i-1].Date.Date)
                yeniSeansBaslangici = true;
        if (yeniSeansBaslangici)
            warmupBaslangicBar = i;
    }
    
    if (warmupAktif && warmupBaslangicBar > 0)
    {
        if ((i - warmupBaslangicBar) < vadeCooldownBar) continue;
        else warmupAktif = false;
    }
    
    if (arefeFlat && i > 0 && dt.Date != V[i-1].Date.Date)
        arefeFlat = false;
    
    // === ÇIKIŞ MANTIĞI (ATR-Based + Double Confirmation) ===
    if (SonYon == "A")
    {
        if (H[i] > extremePrice) extremePrice = H[i];
        float atr = ATR_Exit[i];
        
        // Double Confirmation: N bar ARS altında + mesafe yeterli
        if (C[i] < ARS[i]) belowArsCount++; else belowArsCount = 0;
        float distanceThreshold = (float)(atr * ARS_ATR_Mult * Exit_Confirm_Mult);
        if (belowArsCount >= Exit_Confirm_Bars && (ARS[i] - C[i]) > distanceThreshold)
            Sinyal = "F";
        
        // Take Profit
        float tpLevel = entryPrice + (float)(atr * ATR_TP_Mult);
        if (C[i] >= tpLevel) Sinyal = "F";
        
        // Stop Loss / Trailing
        float initialStop = entryPrice - (float)(atr * ATR_SL_Mult);
        float trailStop = extremePrice - (float)(atr * ATR_Trail_Mult);
        float stopLevel = Math.Max(initialStop, trailStop);
        if (C[i] < stopLevel) Sinyal = "F";
    }
    else if (SonYon == "S")
    {
        if (L[i] < extremePrice) extremePrice = L[i];
        float atr = ATR_Exit[i];
        
        if (C[i] > ARS[i]) aboveArsCount++; else aboveArsCount = 0;
        float distanceThreshold = (float)(atr * ARS_ATR_Mult * Exit_Confirm_Mult);
        if (aboveArsCount >= Exit_Confirm_Bars && (C[i] - ARS[i]) > distanceThreshold)
            Sinyal = "F";
        
        float tpLevel = entryPrice - (float)(atr * ATR_TP_Mult);
        if (C[i] <= tpLevel) Sinyal = "F";
        
        float initialStop = entryPrice + (float)(atr * ATR_SL_Mult);
        float trailStop = extremePrice + (float)(atr * ATR_Trail_Mult);
        float stopLevel = Math.Min(initialStop, trailStop);
        if (C[i] > stopLevel) Sinyal = "F";
    }
    
    // === GİRİŞ MANTIĞI ===
    if (Sinyal == "" && SonYon != "A" && SonYon != "S")
    {
        if (TrendYonu[i] == 1)
        {
            bool yeniZirve = H[i] >= HHV[i-1] && HHV[i] > HHV[i-1];
            bool pozitifMomentum = Momentum[i] > MOMENTUM_THRESHOLD;
            bool mfiOnay = MFI[i] >= MFI_HHV[i-1];
            bool volumeOnay = Lot[i] >= Vol_HHV[i-1] * (float)VOLUME_MULT;
            if (yeniZirve && pozitifMomentum && mfiOnay && volumeOnay) Sinyal = "A";
        }
        else if (TrendYonu[i] == -1)
        {
            bool yeniDip = L[i] <= LLV[i-1] && LLV[i] < LLV[i-1];
            bool negatifMomentum = Momentum[i] < (MOMENTUM_BASE - MOMENTUM_THRESHOLD);
            bool mfiOnay = MFI[i] <= MFI_LLV[i-1];
            bool volumeOnay = Lot[i] >= Vol_HHV[i-1] * (float)VOLUME_MULT;
            if (yeniDip && negatifMomentum && mfiOnay && volumeOnay) Sinyal = "S";
        }
    }
    
    if (Sinyal != "" && SonYon != Sinyal)
    {
        if (Sinyal == "A")
        {
            entryPrice = C[i];
            entryBar = i;
            extremePrice = H[i];
            belowArsCount = 0;
        }
        else if (Sinyal == "S")
        {
            entryPrice = C[i];
            entryBar = i;
            extremePrice = L[i];
            aboveArsCount = 0;
        }
        else if (Sinyal == "F")
        {
            entryPrice = 0;
            extremePrice = 0;
            belowArsCount = 0;
            aboveArsCount = 0;
        }
        
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
    }
}

// --- GÖSTERGELERİ ÇİZ ---
Sistem.Cizgiler[0].Deger = ARS;
Sistem.Cizgiler[0].Aciklama = "ARS";
Sistem.Cizgiler[0].ActiveBool = true;
Sistem.Cizgiler[0].Renk = Color.Yellow;
Sistem.Cizgiler[0].Kalinlik = 2;

Sistem.Cizgiler[1].Deger = HHV;
Sistem.Cizgiler[1].Aciklama = "HHV";

Sistem.Cizgiler[2].Deger = LLV;
Sistem.Cizgiler[2].Aciklama = "LLV";

// ===============================================================================================
// PERFORMANS PANELİ (Detaylı)
// ===============================================================================================
bool GetiriTarihcesiGoster = true;
bool DetayPerformans = true;
string GetiriTarih = "01.01.2024";
float GetiriKayma = 0.0f;

var renk = Color.Black;
DateTime dateBaslangicTarih = DateTime.ParseExact(GetiriTarih, "dd.MM.yyyy", System.Globalization.CultureInfo.CurrentCulture);
if (dateBaslangicTarih < V[0].Date) dateBaslangicTarih = V[0].Date;

Sistem.GetiriHesapla(dateBaslangicTarih.ToString("dd.MM.yyyy"), GetiriKayma);

// Bugünkü getiri kutusu
int ilksatirYy = 240;
var gunluk_getiri = Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] - Sistem.GetiriKZGun[Sistem.GetiriKZGun.Count - 1];
var kzbugunx = gunluk_getiri.ToString("0.0");
if (gunluk_getiri > 0) renk = Color.Green; else if (gunluk_getiri < 0) renk = Color.Red;

Sistem.Dortgen(1, 10, ilksatirYy - 5, 90, 25, renk, Color.Black, Color.White);
Sistem.GradientYaziEkle("Bugün", 1, 15, ilksatirYy, Color.White, Color.White, "Tahoma", 8);
Sistem.GradientYaziEkle(kzbugunx, 1, 60, ilksatirYy, Color.Yellow, Color.DarkOrange, "Tahoma", 8);

if (Sistem.Parametreler[3] == "X")
{
    int ilksatirY = 33;
    var Sure = ((DateTime.Now - dateBaslangicTarih).TotalDays / 30.4);
    var SureTxt = Sure.ToString("0.0");
    var kzSure = Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1].ToString("0.0");
    var kzbugun = (Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] - Sistem.GetiriKZGun[Sistem.GetiriKZGun.Count - 1]).ToString("0.0");
    var yuzde_kz = (Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] * 100.0f) / O[0];
    var kzSure_yuzde = "  %" + yuzde_kz.ToString("0.0");

    var kzbuay = Sistem.GetiriBuAy.ToString("0.0");
    var kz30 = Sistem.GetiriBirAy.ToString("0.0");
    string ToplamIslem = Sistem.GetiriToplamIslem.ToString("0");
    string OrtalamaIslem = (((double)Sistem.GetiriToplamIslem) / Sure).ToString("0");
    var KarliIslemOran = Sistem.GetiriKarIslemOran.ToString("0.00");
    var MutluGun = Sistem.GetiriMutluGun.ToString();
    var MutsuzGun = Sistem.GetiriMutsuzGun.ToString();
    Sistem.GetiriMaxDDHesapla(GetiriTarih, DateTime.Now.ToString("dd.MM.yyyy"));
    var MaxDD = Sistem.GetiriMaxDD.ToString("0.0");
    var MaxDDTarihi = Sistem.GetiriMaxDDTarih.ToString("dd.MM.yyyy");
    var ProfitFactor = Sistem.ProfitFactor.ToString("0.00");

    // Getiri çizgileri
    Sistem.Cizgiler[3].Deger = Sistem.GetiriKZGun;
    Sistem.Cizgiler[3].Aciklama = "Gün KZ";
    Sistem.Cizgiler[3].ActiveBool = true;
    
    Sistem.Cizgiler[4].Deger = Sistem.GetiriKZGunSonu;
    Sistem.Cizgiler[4].Aciklama = "Gün Sonu KZ";
    Sistem.Cizgiler[4].ActiveBool = true;
    
    Sistem.DolguEkle(3, 4, Color.Red, Color.Green);
    
    Sistem.Cizgiler[5].Deger = Sistem.GetiriKZAy;
    Sistem.Cizgiler[5].Aciklama = "Aylık Getiri";
    Sistem.Cizgiler[5].ActiveBool = true;

    if (GetiriTarihcesiGoster)
    {
        // Geçmiş dönem getiri hesaplamaları
        var Date5Ay = DateTime.Now.AddDays(-5);
        int Date5AyBarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= Date5Ay) { Date5AyBarNo = i; break; }
        var kz5 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date5AyBarNo]).ToString("0.0");

        var Date60 = DateTime.Now.AddDays(-60);
        int Date60BarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= Date60) { Date60BarNo = i; break; }
        var kz60 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date60BarNo]).ToString("0.0");

        var Date90 = DateTime.Now.AddDays(-90);
        int Date90BarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= Date90) { Date90BarNo = i; break; }
        var kz90 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date90BarNo]).ToString("0.0");

        var Date180 = DateTime.Now.AddDays(-180);
        int Date180BarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= Date180) { Date180BarNo = i; break; }
        var kz180 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date180BarNo]).ToString("0.0");

        var DateYilBasi = new DateTime(DateTime.Now.Year, 1, 1);
        int DateYilBasiBarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= DateYilBasi) { DateYilBasiBarNo = i; break; }
        var kzBuYil = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[DateYilBasiBarNo]).ToString("0.0");

        var Date1Yil = DateTime.Now.AddYears(-1);
        int Date1YilBarNo = 0;
        for (int i = V.Count - 1; i > 0; i--)
            if (V[i].Date <= Date1Yil) { Date1YilBarNo = i; break; }
        var kz1Yil = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date1YilBarNo]).ToString("0.0");

        string Labels = SureTxt + " Ay" + Environment.NewLine +
                        "Bugün" + Environment.NewLine +
                        "Bu Hafta" + Environment.NewLine +
                        "Bu Ay" + Environment.NewLine +
                        "30 Gün" + Environment.NewLine +
                        "60 Gün" + Environment.NewLine +
                        "90 Gün" + Environment.NewLine +
                        "180 Gün" + Environment.NewLine +
                        "Bu Yıl" + Environment.NewLine +
                        "Son 1 Yıl";

        string Results = kzSure + kzSure_yuzde + Environment.NewLine +
                         kzbugun + Environment.NewLine +
                         kz5 + Environment.NewLine +
                         kzbuay + Environment.NewLine +
                         kz30 + Environment.NewLine +
                         kz60 + Environment.NewLine +
                         kz90 + Environment.NewLine +
                         kz180 + Environment.NewLine +
                         kzBuYil + Environment.NewLine +
                         kz1Yil;

        Sistem.Dortgen(2, 10, ilksatirY - 8, 230, 180, Color.Black, Color.Black, Color.White);
        Sistem.GradientYaziEkle(Labels, 2, 20, ilksatirY, Color.White, Color.White, "Tahoma", 10);
        Sistem.GradientYaziEkle(Results, 2, 90, ilksatirY, Color.Yellow, Color.DarkOrange, "Tahoma", 10);
    }

    if (DetayPerformans)
    {
        string Labels2 = "İşlem / Ortalama" + Environment.NewLine +
                         "Karlı İşlem Oranı" + Environment.NewLine +
                         "Profit Factor" + Environment.NewLine +
                         "Mutlu Gün" + Environment.NewLine +
                         "Mutsuz Gün" + Environment.NewLine +
                         "MaxDD" + Environment.NewLine +
                         "MaxDD Tarihi";

        string Results2 = ToplamIslem + " / " + OrtalamaIslem + Environment.NewLine +
                          "%" + KarliIslemOran + Environment.NewLine +
                          ProfitFactor + Environment.NewLine +
                          MutluGun + Environment.NewLine +
                          MutsuzGun + Environment.NewLine +
                          MaxDD + Environment.NewLine +
                          MaxDDTarihi;

        Sistem.Dortgen(2, 250, ilksatirY - 8, 220, 130, Color.Black, Color.Black, Color.White);
        Sistem.GradientYaziEkle(Labels2, 2, 260, ilksatirY, Color.White, Color.White, "Tahoma", 10);
        Sistem.GradientYaziEkle(Results2, 2, 385, ilksatirY, Color.Yellow, Color.DarkOrange, "Tahoma", 10);
    }
}
