// ===============================================================================================
// STRATEJİ 2: ARS TREND TAKİP SİSTEMİ
// ===============================================================================================
// Sembol: VIP'VIP-X030
// Periyot: 5 dakika
// Vade Tipi: ENDEKS
// Oluşturma: 2026-02-01 02:00
// ===============================================================================================

// --- ARS PARAMETRELER ---
var P_ARS_EMA = 3;
var P_ARS_ATR_Period = 14;
var P_ARS_ATR_Mult = 0.5;

// --- GİRİŞ PARAMETRELER ---
var P_Momentum = 8;
var P_Breakout_Short = 8;
var P_Breakout_Mid = 20;
var P_Breakout_Long = 50;
var P_MFI_Period = 14;
var P_MFI_HHV = 14;
var P_Volume_Period = 20;

// --- ÇIKIŞ PARAMETRELER ---
var P_ATR_Exit = 14;
var P_ATR_SL_Mult = 2.0;
var P_ATR_TP_Mult = 3.0;
var P_ATR_Trail_Mult = 1.5;
var P_Exit_Confirm_Bars = 2;
var P_Exit_Confirm_Mult = 1.0;

// --- VERİ ---
var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var C = Sistem.GrafikFiyatSec("Kapanis");
var T = Sistem.GrafikFiyatSec("Tipik");
var Vol = Sistem.GrafikFiyatSec("Hacim");

// --- ARS HESAPLAMA ---
var ATR = Sistem.AverageTrueRange(P_ARS_ATR_Period);
var ARS_EMA = Sistem.MA(T, "Exp", P_ARS_EMA);
var ARS = Sistem.Liste(0);

for (int i = 1; i < V.Count; i++)
{
    float dinamikK = (ATR[i] / ARS_EMA[i]) * (float)P_ARS_ATR_Mult;
    dinamikK = Math.Max(0.002f, Math.Min(0.020f, dinamikK));
    
    float altBand = ARS_EMA[i] * (1 - dinamikK);
    float ustBand = ARS_EMA[i] * (1 + dinamikK);
    
    if (altBand > ARS[i - 1])
        ARS[i] = altBand;
    else if (ustBand < ARS[i - 1])
        ARS[i] = ustBand;
    else
        ARS[i] = ARS[i - 1];
}

// --- TREND ---
var TrendYonu = Sistem.Liste(0);
for (int i = 1; i < V.Count; i++)
{
    if (C[i] > ARS[i]) TrendYonu[i] = 1;
    else if (C[i] < ARS[i]) TrendYonu[i] = -1;
    else TrendYonu[i] = TrendYonu[i-1];
}

// --- GİRİŞ İNDİKATÖRLER ---
var Momentum = Sistem.Momentum(P_Momentum);
var HHV_Short = Sistem.HHV(P_Breakout_Short);
var LLV_Short = Sistem.LLV(P_Breakout_Short);
var HHV_Mid = Sistem.HHV(P_Breakout_Mid);
var LLV_Mid = Sistem.LLV(P_Breakout_Mid);

// MFI
var MFI = Sistem.MFI(P_MFI_Period);
var MFI_HHV = Sistem.HHV(MFI, P_MFI_HHV);

// --- ÇIKIŞ İNDİKATÖR ---
var ATR_Exit = Sistem.AverageTrueRange(P_ATR_Exit);

// --- SİNYAL ÜRETİMİ ---
var SonYon = "";
float girisFiyat = 0;
float stopFiyat = 0;
float hedefFiyat = 0;
float trailStop = 0;
int trendDonusCounter = 0;

int warmup = Math.Max(P_Breakout_Long, Math.Max(P_ARS_ATR_Period, P_MFI_HHV)) + 5;

for (int i = warmup; i < V.Count; i++)
{
    var Sinyal = "";
    
    // === ÇIKIŞ MANTIĞI ===
    if (SonYon == "A")
    {
        // Stop Loss
        if (L[i] <= stopFiyat) Sinyal = "F";
        // Take Profit
        else if (H[i] >= hedefFiyat) Sinyal = "F";
        // Trailing Stop
        else if (C[i] < trailStop) Sinyal = "F";
        // Trend dönüşü (çift teyit)
        else if (TrendYonu[i] == -1)
        {
            trendDonusCounter++;
            if (trendDonusCounter >= P_Exit_Confirm_Bars) Sinyal = "F";
        }
        else trendDonusCounter = 0;
        
        // Trail güncelle
        if (Sinyal == "")
        {
            float yeniTrail = H[i] - ATR_Exit[i] * (float)P_ATR_Trail_Mult;
            if (yeniTrail > trailStop) trailStop = yeniTrail;
        }
    }
    else if (SonYon == "S")
    {
        if (H[i] >= stopFiyat) Sinyal = "F";
        else if (L[i] <= hedefFiyat) Sinyal = "F";
        else if (C[i] > trailStop) Sinyal = "F";
        else if (TrendYonu[i] == 1)
        {
            trendDonusCounter++;
            if (trendDonusCounter >= P_Exit_Confirm_Bars) Sinyal = "F";
        }
        else trendDonusCounter = 0;
        
        if (Sinyal == "")
        {
            float yeniTrail = L[i] + ATR_Exit[i] * (float)P_ATR_Trail_Mult;
            if (yeniTrail < trailStop) trailStop = yeniTrail;
        }
    }
    
    // === GİRİŞ MANTIĞI ===
    if (Sinyal == "" && SonYon != "A" && SonYon != "S")
    {
        bool mfiBreakout = MFI[i] >= MFI_HHV[i-1];
        
        if (TrendYonu[i] == 1)
        {
            bool yeniZirve = H[i] >= HHV_Short[i-1];
            bool momentum = Momentum[i] > 100;
            
            if (yeniZirve && momentum && mfiBreakout)
            {
                Sinyal = "A";
                girisFiyat = C[i];
                stopFiyat = girisFiyat - ATR_Exit[i] * (float)P_ATR_SL_Mult;
                hedefFiyat = girisFiyat + ATR_Exit[i] * (float)P_ATR_TP_Mult;
                trailStop = stopFiyat;
                trendDonusCounter = 0;
            }
        }
        else if (TrendYonu[i] == -1)
        {
            bool yeniDip = L[i] <= LLV_Short[i-1];
            bool momentum = Momentum[i] < 100;
            
            if (yeniDip && momentum && mfiBreakout)
            {
                Sinyal = "S";
                girisFiyat = C[i];
                stopFiyat = girisFiyat + ATR_Exit[i] * (float)P_ATR_SL_Mult;
                hedefFiyat = girisFiyat - ATR_Exit[i] * (float)P_ATR_TP_Mult;
                trailStop = stopFiyat;
                trendDonusCounter = 0;
            }
        }
    }
    
    if (Sinyal != "" && Sinyal != SonYon)
    {
        Sistem.Yon[i] = Sinyal;
        SonYon = Sinyal;
    }
}

// --- GÖSTERGELERİ ÇİZ ---
Sistem.Cizgiler[0].Deger = ARS;
Sistem.Cizgiler[0].Aciklama = "ARS";
Sistem.Cizgiler[1].Deger = HHV_Short;
Sistem.Cizgiler[1].Aciklama = "HHV";
Sistem.Cizgiler[2].Deger = LLV_Short;
Sistem.Cizgiler[2].Aciklama = "LLV";
