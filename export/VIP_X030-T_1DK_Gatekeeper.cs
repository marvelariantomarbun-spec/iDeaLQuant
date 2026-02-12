// ===============================================================================================
// STRATEJI 1: GATEKEEPER (MACDV + ARS + ADX + NETLOT)
// ===============================================================================================
// Sembol: VIP_X030-T
// Periyot: 1 dakika
// Vade Tipi: ENDEKS
// Oluşturma: 2026-02-12 17:46
// ===============================================================================================

// --- PARAMETRELER ---
var MIN_ONAY_SKORU = 2;
var CIKIS_HASSASIYETI = 3;
var ARS_PERIYOT = 30;
var ARS_K = 0.005;
var ADX_PERIOD = 50;
var ADX_ESIK = 40.0f;
var NETLOT_ESIK = 12.5f;
var NETLOT_PERIOD = 45;
var MACDV_K = 90;
var MACDV_U = 100;
var MACDV_SIG = 35;
var MACDV_ESIK = 10.001f;

// --- YATAY FİLTRE PARAMETRELERİ ---
var YATAY_ARS_BARS = 75;
var ARS_MESAFE_ESIK = 0.25f;
var YATAY_ADX_ESIK = 20.0f;
var BB_PERIOD = 400;
var BB_STD = 1.5f;
var BB_WIDTH_MULT = 0.5f;
var BB_AVG_PERIOD = 300;
var FILTRE_SKOR_ESIK = 2;

// --- VADE TİPİ ---
string VadeTipi = "ENDEKS";


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

// Optimizasyon için tüm yıllar
DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] { "01.01","04.23","05.01","05.19","07.15","08.30","10.29" };

var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var T = Sistem.GrafikFiyatSec("Tipik");

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

// --- 1. ARS ---
var ARS_EMA = Sistem.MA(T, "Exp", ARS_PERIYOT);
var ARS = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {
    float altBand = (float)(ARS_EMA[i] * (1 - ARS_K));
    float ustBand = (float)(ARS_EMA[i] * (1 + ARS_K));
    if (altBand > ARS[i - 1]) ARS[i] = altBand;
    else if (ustBand < ARS[i - 1]) ARS[i] = ustBand;
    else ARS[i] = ARS[i - 1];
}

// --- 2. MACDV ---
var EMA_S = Sistem.MA(C, "Exp", MACDV_K);
var EMA_L = Sistem.MA(C, "Exp", MACDV_U);

var TR_List = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {
    float hl = H[i] - L[i];
    float hc = Math.Abs(H[i] - C[i-1]);
    float lc = Math.Abs(L[i] - C[i-1]);
    TR_List[i] = Math.Max(hl, Math.Max(hc, lc));
}
var ATRe = Sistem.MA(TR_List, "Exp", MACDV_U);

var MACDV = Sistem.Liste(0);
for (int i = 0; i < Sistem.BarSayisi; i++) {
    if (ATRe[i] != 0)
        MACDV[i] = ((EMA_S[i] - EMA_L[i]) / ATRe[i]) * 100;
}
var MACDV_Sinyal = Sistem.MA(MACDV, "Exp", MACDV_SIG);

// --- 3. YATAY FILTRE ---
var ARS_Degisim = Sistem.Liste(0);
for (int i = YATAY_ARS_BARS; i < Sistem.BarSayisi; i++) {
    bool arsAyni = true;
    for (int j = 1; j <= YATAY_ARS_BARS; j++)
        if (ARS[i] != ARS[i - j]) { arsAyni = false; break; }
    ARS_Degisim[i] = arsAyni ? 0 : 1;
}

var ARS_Mesafe = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++)
    ARS_Mesafe[i] = Math.Abs(C[i] - ARS[i]) / ARS[i] * 100;

var ADX14 = Sistem.ADX(ADX_PERIOD);

var BBUp = Sistem.BollingerUp("Simple", BB_PERIOD, BB_STD);
var BBDown = Sistem.BollingerDown("Simple", BB_PERIOD, BB_STD);
var BBMid = Sistem.BollingerMid("Simple", BB_PERIOD, BB_STD);
var BBWidth = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++)
    if (BBMid[i] != 0) BBWidth[i] = ((BBUp[i] - BBDown[i]) / BBMid[i]) * 100;
var BBWidth_Avg = Sistem.MA(BBWidth, "Simple", BB_AVG_PERIOD);

var YatayFiltre = Sistem.Liste(0);
for (int i = BB_AVG_PERIOD; i < Sistem.BarSayisi; i++) {
    int skor = 0;
    if (ARS_Degisim[i] == 1) skor++;
    if (ARS_Mesafe[i] > ARS_MESAFE_ESIK) skor++;
    if (ADX14[i] > YATAY_ADX_ESIK) skor++;
    if (BBWidth[i] > BBWidth_Avg[i] * BB_WIDTH_MULT) skor++;
    YatayFiltre[i] = (skor >= FILTRE_SKOR_ESIK) ? 1 : 0;
}

// --- 4. NET HACİM ---
var NetLot = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {
    float barHacim = (H[i] - L[i]) > 0 ? (C[i] - O[i]) / (H[i] - L[i]) : 0;
    NetLot[i] = barHacim * 100;
}
var NetLot_MA = Sistem.MA(NetLot, "Simple", NETLOT_PERIOD);


// --- SINYAL ---
for (int i = 1; i < V.Count; i++) Sistem.Yon[i] = "";
var SonYon = "";

int vadeCooldownBar = Math.Max(ARS_PERIYOT, Math.Max(ADX_PERIOD, Math.Max(MACDV_K, MACDV_U))) + 10;
int warmupBars = Math.Max(50, vadeCooldownBar);
int warmupBaslangicBar = -999;
bool warmupAktif = false;
bool arefeFlat = false;

for (int i = warmupBars; i < Sistem.BarSayisi; i++)
{
    var Sinyal = "";
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
    
    // --- SKORLAMA ---
    int longScore = 0;
    int shortScore = 0;

    if (C[i] > ARS[i]) longScore++; else if (C[i] < ARS[i]) shortScore++;
    if (MACDV[i] > (MACDV_Sinyal[i] + MACDV_ESIK)) longScore++; else if (MACDV[i] < (MACDV_Sinyal[i] - MACDV_ESIK)) shortScore++;
    if (NetLot_MA[i] > NETLOT_ESIK) longScore++; else if (NetLot_MA[i] < -NETLOT_ESIK) shortScore++;
    if (ADX14[i] > ADX_ESIK) { longScore++; shortScore++; }


    // --- ÇIKIŞ MANTIĞI ---
    if (SonYon == "A") {
        if (C[i] < ARS[i] || shortScore >= CIKIS_HASSASIYETI) Sinyal = "F";
    }
    else if (SonYon == "S") {
        if (C[i] > ARS[i] || longScore >= CIKIS_HASSASIYETI) Sinyal = "F";
    }
    
    // --- GİRİŞ MANTIĞI ---
    if (Sinyal == "" && SonYon != "A" && SonYon != "S") {
        if (YatayFiltre[i] == 1) {
            if (longScore >= MIN_ONAY_SKORU && shortScore < 2) Sinyal = "A";
            else if (shortScore >= MIN_ONAY_SKORU && longScore < 2) Sinyal = "S";
        }
    }
    
    // --- POZİSYON GÜNCELLEME ---
    if (Sinyal != "" && SonYon != Sinyal) {
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
    }
}

// --- CIZIMLER ---
Sistem.Cizgiler[0].Deger = ARS;
Sistem.Cizgiler[0].Aciklama = "ARS";
Sistem.Cizgiler[0].ActiveBool = true;
Sistem.Cizgiler[0].Renk = Color.Yellow;
Sistem.Cizgiler[0].Kalinlik = 2;

Sistem.Cizgiler[1].Deger = MACDV;
Sistem.Cizgiler[1].Aciklama = "MACDV";
Sistem.Cizgiler[1].ActiveBool = false;

var son = Sistem.BarSayisi - 1;
string info = "MACDV: " + Sistem.SayiYuvarla(MACDV[son], 2) + " Sig: " + Sistem.SayiYuvarla(MACDV_Sinyal[son], 2);
Sistem.Dortgen(1, 100, 300, 200, 50, Color.Black, Color.White, Color.Black);
Sistem.YaziEkle(info, 1, 15, 35, Color.Yellow, "Tahoma", 10);
