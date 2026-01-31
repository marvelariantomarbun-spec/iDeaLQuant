// ===============================================================================================
// STRATEJİ 1: YATAY FİLTRE + SCORING SİSTEMİ
// ===============================================================================================
// Sembol: VIP'VIP-X030
// Periyot: 5 dakika
// Vade Tipi: ENDEKS
// Oluşturma: 2026-02-01 02:00
// ===============================================================================================

// --- PARAMETRELER ---
var P_ADX_Period = 14;
var P_ADX_Threshold = 25;
var P_MACDV_Fast = 12;
var P_MACDV_Slow = 26;
var P_MACDV_Signal = 9;
var P_EMA_Short = 10;
var P_EMA_Long = 50;
var P_ATR_Period = 14;
var P_Volume_Period = 20;
var P_Long_Score_Threshold = 3;
var P_Short_Score_Threshold = 3;

// --- VERİ ---
var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var Vol = Sistem.GrafikFiyatSec("Hacim");

// --- İNDİKATÖRLER ---
var ADX = Sistem.ADX(P_ADX_Period);
var EMA_Short = Sistem.MA(C, "Exp", P_EMA_Short);
var EMA_Long = Sistem.MA(C, "Exp", P_EMA_Long);
var ATR = Sistem.AverageTrueRange(P_ATR_Period);
var Vol_SMA = Sistem.MA(Vol, "Simple", P_Volume_Period);

// MACD-V hesaplama
var MACD_Line = Sistem.Liste(0);
var MACD_Signal = Sistem.Liste(0);
var MACD_Hist = Sistem.Liste(0);
var EMA_Fast = Sistem.MA(C, "Exp", P_MACDV_Fast);
var EMA_Slow = Sistem.MA(C, "Exp", P_MACDV_Slow);

for (int i = 0; i < V.Count; i++)
{
    if (ATR[i] > 0)
        MACD_Line[i] = (EMA_Fast[i] - EMA_Slow[i]) / ATR[i];
}
var MACD_Sig_EMA = Sistem.MA(MACD_Line, "Exp", P_MACDV_Signal);
for (int i = 0; i < V.Count; i++)
{
    MACD_Signal[i] = MACD_Sig_EMA[i];
    MACD_Hist[i] = MACD_Line[i] - MACD_Signal[i];
}

// --- YATAY FİLTRE ---
var YatayFiltre = Sistem.Liste(0);
for (int i = 0; i < V.Count; i++)
{
    YatayFiltre[i] = (ADX[i] < P_ADX_Threshold) ? 1 : 0;
}

// --- SCORING ---
var LongScore = Sistem.Liste(0);
var ShortScore = Sistem.Liste(0);

for (int i = 1; i < V.Count; i++)
{
    int ls = 0, ss = 0;
    
    // EMA Trend
    if (EMA_Short[i] > EMA_Long[i]) ls++; else ss++;
    
    // MACD-V
    if (MACD_Hist[i] > 0) ls++; else ss++;
    if (MACD_Hist[i] > MACD_Hist[i-1]) ls++; else ss++;
    
    // Volume
    if (Vol[i] > Vol_SMA[i]) { ls++; ss++; }
    
    // Momentum
    if (C[i] > C[i-1]) ls++; else ss++;
    
    LongScore[i] = ls;
    ShortScore[i] = ss;
}

// --- SİNYAL ÜRETİMİ ---
var SonYon = "";
for (int i = 1; i < V.Count; i++)
{
    var Sinyal = "";
    
    // Yatay filtre aktif ve skor yeterli
    if (YatayFiltre[i] == 1)
    {
        if (LongScore[i] >= P_Long_Score_Threshold && SonYon != "A")
            Sinyal = "A";
        else if (ShortScore[i] >= P_Short_Score_Threshold && SonYon != "S")
            Sinyal = "S";
    }
    else
    {
        // Yatay değilse flat
        if (SonYon == "A" || SonYon == "S")
            Sinyal = "F";
    }
    
    if (Sinyal != "" && Sinyal != SonYon)
    {
        Sistem.Yon[i] = Sinyal;
        SonYon = Sinyal;
    }
}

// --- GÖSTERGELERİ ÇİZ ---
Sistem.Cizgiler[0].Deger = ADX;
Sistem.Cizgiler[0].Aciklama = "ADX";
Sistem.Cizgiler[1].Deger = LongScore;
Sistem.Cizgiler[1].Aciklama = "Long Score";
Sistem.Cizgiler[2].Deger = ShortScore;
Sistem.Cizgiler[2].Aciklama = "Short Score";
