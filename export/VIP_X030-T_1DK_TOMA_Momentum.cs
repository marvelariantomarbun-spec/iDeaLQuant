// ===============================================================================================
// STRATEJI 4: TOMA + MOMENTUM (Karma Sistem)
// ===============================================================================================
// Sembol: VIP_X030-T
// Periyot: 1 dakika
// Vade Tipi: ENDEKS
// Olusturma: 2026-02-22 01:37
// ===============================================================================================

// --- VADE TİPİ ---
string VadeTipi = "ENDEKS";

// --- PARAMETRELER ---
var MOM_PERIOD = 1400;
var MOM_UPPER = 100.0f;
var MOM_LOWER = 99.0f;
var TRIX_PERIOD = 110;
var TRIX_LB1 = 40;
var TRIX_LB2 = 20;
var HH_LL_PERIOD = 375;
var HHV2_PERIOD = 90;
var LLV2_PERIOD = 405;
var HHV3_PERIOD = 90;
var LLV3_PERIOD = 30;
var TOMA_PERIOD = 3;
var TOMA_OPT = 0.9f;

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

var V = Sistem.GrafikVerileri;
var C = Sistem.GrafikFiyatSec("Kapanis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");

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

// --- INDIKATORLER ---
var TOMA_Line = Sistem.TOMA(TOMA_PERIOD, TOMA_OPT);
// Sistem.TOMA sadece çizgiyi döner, trend yönünü değil. Kapanış ile karşılaştıracağız.

var HH1 = Sistem.HHV(HH_LL_PERIOD, "Yuksek");
var LL1 = Sistem.LLV(HH_LL_PERIOD, "Dusuk");

var HH2 = Sistem.HHV(HHV2_PERIOD, "Yuksek");
var LL2 = Sistem.LLV(LLV2_PERIOD, "Dusuk");

var HH3 = Sistem.HHV(HHV3_PERIOD, "Yuksek");
var LL3 = Sistem.LLV(LLV3_PERIOD, "Dusuk");

var MOM1 = Sistem.Momentum(MOM_PERIOD);
var TRIX1 = Sistem.TRIX(TRIX_PERIOD);
var TRIX2 = Sistem.TRIX(TRIX_PERIOD);

// --- LOOP & SINYAL ---
var SonYon = "";
var Sinyal = "";
double IslemFiyati = 0.0;
var Pos = 0;

for (int i = 1; i < V.Count; i++) Sistem.Yon[i] = "";

int vadeCooldownBar = Math.Max(MOM_PERIOD, TRIX_PERIOD + Math.Max(TRIX_LB1, TRIX_LB2)) + 10;
int warmupBars = Math.Max(200, vadeCooldownBar);
int warmupBaslangicBar = -999;
bool warmupAktif = false;
bool arefeFlat = false;

for (int i = warmupBars; i < V.Count; i++)
{
    Sinyal = "";
    var dt = V[i].Date;
    var t = dt.TimeOfDay;
    
    // --- VADE/TATİL KONTROLLERİ ---
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
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }
    
    if (Sinyal == "F") {
        if (SonYon != Sinyal) { Sistem.Yon[i] = Sinyal; SonYon = Sinyal; Pos = 0; }
        continue;
    }
    
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0))) continue;

    if (warmupAktif && warmupBaslangicBar == -999) {
        bool yeniSeans = false;
        if (aksamSeansi && i>0 && V[i-1].Date.TimeOfDay < new TimeSpan(19,0,0)) yeniSeans = true;
        if (gunSeansi && i>0 && dt.Date != V[i-1].Date.Date) yeniSeans = true;
        if (yeniSeans) warmupBaslangicBar = i;
    }
    if (warmupAktif && warmupBaslangicBar > 0) {
        if ((i - warmupBaslangicBar) < vadeCooldownBar) continue;
        else warmupAktif = false;
    }
    if (arefeFlat && i>0 && dt.Date != V[i-1].Date.Date) arefeFlat = false;


    // --- STRATEJİ MANTIĞI ---
    
    // Kural 1: MOM > ÜST SINIR (101.5)
    if (MOM1[i] > MOM_UPPER)
    {
        if (HH2[i] > HH2[i-1] && TRIX1[i] < TRIX1[i-TRIX_LB1] && TRIX1[i] > TRIX1[i-1]) Sinyal = "A"; 
        if (LL2[i] < LL2[i-1] && TRIX1[i] > TRIX1[i-TRIX_LB1] && TRIX1[i] < TRIX1[i-1]) Sinyal = "S"; 
    }
    
    // Kural 2: MOM < ALT SINIR (98)
    if (MOM1[i] < MOM_LOWER)
    {
        if (HH3[i] > HH3[i-1] && TRIX2[i] < TRIX2[i-TRIX_LB2] && TRIX2[i] > TRIX2[i-1]) Sinyal = "A"; 
        if (LL3[i] < LL3[i-1] && TRIX2[i] > TRIX2[i-TRIX_LB2] && TRIX2[i] < TRIX2[i-1]) Sinyal = "S"; 
    }
    
    // Kural 3: TOMA + HHV/LLV (Ana Trend - Öncelikli, önceki sinyalleri ezer)
    if (HH1[i] > HH1[i-1] && C[i] > TOMA_Line[i]) Sinyal = "A";
    if (LL1[i] < LL1[i-1] && C[i] < TOMA_Line[i]) Sinyal = "S";

    // --- POZİSYON GÜNCELLEME ---
    if (Sinyal != "" && SonYon != Sinyal)
    {
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
        IslemFiyati = C[i];
        if (Sinyal == "A") Pos = 1;
        else if (Sinyal == "S") Pos = -1;
        else Pos = 0;
    }
}

// --- ÇİZİMLER ---
Sistem.Cizgiler[0].Deger = TOMA_Line;
Sistem.Cizgiler[0].Aciklama = "TOMA";
Sistem.Cizgiler[0].Renk = Color.Blue;
Sistem.Cizgiler[0].Kalinlik = 2;

Sistem.Cizgiler[1].Deger = HH1;
Sistem.Cizgiler[1].Aciklama = "HH1";
Sistem.Cizgiler[1].ActiveBool = false;

Sistem.Cizgiler[2].Deger = LL1;
Sistem.Cizgiler[2].Aciklama = "LL1";
Sistem.Cizgiler[2].ActiveBool = false;


// ===============================================================================================
// PERFORMANS PANELİ (STANDART)
// ===============================================================================================
bool GetiriTarihcesiGoster = true;
bool DetayPerformans = true;
string GetiriTarih = "01.01.2024";
float GetiriKayma = 0.0f;

//-----------------------------------------------
var renk = Color.Black;
var Grafikler = Sistem.GrafikVerileri;

DateTime dateBaslangicTarih = (DateTime.ParseExact(GetiriTarih, "dd.MM.yyyy", System.Globalization.CultureInfo.CurrentCulture) > Grafikler[0].Date) ? (DateTime.ParseExact(GetiriTarih, "dd.MM.yyyy", System.Globalization.CultureInfo.CurrentCulture)) : Grafikler[0].Date;
Sistem.GetiriHesapla(dateBaslangicTarih.ToString("dd.MM.yyyy"), GetiriKayma); 

int ilksatirYy = 240;
var gunluk_getiri = Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] - Sistem.GetiriKZGun[Sistem.GetiriKZGun.Count - 1];
var kzbugunx      = gunluk_getiri.ToString("0.0");
string Labelsx    =  "Bugün" + Environment.NewLine ;
string Resultsx   = kzbugunx + Environment.NewLine ;

if ( gunluk_getiri > 0 ) renk = Color.Green; else if ( gunluk_getiri < 0 ) renk = Color.Red;

Sistem.Dortgen(1, 10, ilksatirYy - 5, 90, 25, renk, Color.Black, Color.White);
Sistem.GradientYaziEkle(Labelsx, 1, 15, ilksatirYy, Color.White, Color.White, "Tahoma", 8);
Sistem.GradientYaziEkle(Resultsx, 1, 60, ilksatirYy, Color.Yellow, Color.DarkOrange, "Tahoma", 8);
//-----------------------------------------------

// Parametre 3 "X" ise (Panelde 4. Satır) Performans Çizgilerini Göster (Overwrite)
if (Sistem.Parametreler[3] == "X")
{
    int ilksatirY = 33;
    var Sure = ((DateTime.Now - dateBaslangicTarih).TotalDays / 30.4);
    var SureTxt = Sure.ToString("0.0");
    var kzSure = Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1].ToString("0.0");
    var kzbugun = (Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] - Sistem.GetiriKZGun[Sistem.GetiriKZGun.Count - 1]).ToString("0.0");
    var yuzde_kz =  ( Sistem.GetiriKZGunSonu[Sistem.GetiriKZGunSonu.Count - 1] * 100.0f ) / O[0];
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

    // Çizgileri Ezme (Overlay Modu)
    Sistem.Cizgiler[0].Deger = Sistem.GetiriKZGun; 
    Sistem.Cizgiler[0].Aciklama = "Gün KZ"; 
    Sistem.Cizgiler[0].ActiveBool = true;
    
    Sistem.Cizgiler[1].Deger = Sistem.GetiriKZGunSonu;
    Sistem.Cizgiler[1].Aciklama = "Gün Sonu KZ"; 
    Sistem.Cizgiler[1].ActiveBool = true;
    
    Sistem.DolguEkle(0, 1, Color.Red, Color.Green);
    
    Sistem.Cizgiler[2].Deger = Sistem.GetiriKZAy; 
    Sistem.Cizgiler[2].Aciklama =  "Aylık Getiri"; 
    Sistem.Cizgiler[2].ActiveBool = true;

    if (GetiriTarihcesiGoster)
    {
        var Date5Ay = DateTime.Now.AddDays(-5);
        var Date5AyBarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= Date5Ay) { Date5AyBarNo = i; break; }
        }
        var kz5 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date5AyBarNo]).ToString("0.0");

        var Date60 = DateTime.Now.AddDays(-60);
        var Date60BarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= Date60) { Date60BarNo = i; break; }
        }
        var kz60 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date60BarNo]).ToString("0.0");

        var Date90 = DateTime.Now.AddDays(-90);
        var Date90BarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= Date90) { Date90BarNo = i; break; }
        }
        var kz90 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date90BarNo]).ToString("0.0");

        var Date180 = DateTime.Now.AddDays(-180);
        var Date180BarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= Date180) { Date180BarNo = i; break; }
        }
        var kz180 = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[Date180BarNo]).ToString("0.0");

        // Yıl başı getirisi hesaplama
        var DateYilBasi = new DateTime(DateTime.Now.Year, 1, 1);
        var DateYilBasiBarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= DateYilBasi) { DateYilBasiBarNo = i; break; }
        }
        var kzBuYil = (Sistem.GetiriKZ[Sistem.GetiriKZ.Count - 1] - Sistem.GetiriKZ[DateYilBasiBarNo]).ToString("0.0");

        // Son 1 yıl getirisi hesaplama
        var Date1Yil = DateTime.Now.AddYears(-1);
        var Date1YilBarNo = 0;
        for (int i = Grafikler.Count - 1; i > 0; i--)
        {
            if (Grafikler[i].Date <= Date1Yil) { Date1YilBarNo = i; break; }
        }
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

        string Results = kzSure + kzSure_yuzde+  Environment.NewLine +
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
        string Labels2 = "İslem / Ortalama" + Environment.NewLine +
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

