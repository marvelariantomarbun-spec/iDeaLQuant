# -*- coding: utf-8 -*-
"""
IdealData Export Module
-----------------------
Optimizasyon sonuçlarını IdealData uyumlu strateji ve robot kodlarına çevirir.

Kullanım:
    exporter = IdealDataExporter(symbol="VIP'VIP-X030", period="5")
    exporter.export_strategy1(params1, "ENDEKS")
    exporter.export_strategy2(params2, "ENDEKS")
    exporter.export_combined_robot()
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


class IdealDataExporter:
    """Strateji ve robot kodu export sınıfı."""
    
    def __init__(
        self,
        symbol: str = "VIP'VIP-X030",
        period: str = "5",
        output_dir: str = None
    ):
        """
        Args:
            symbol: IdealData sembol adı (örn: VIP'VIP-X030)
            period: Grafik periyodu (1, 5, 15, 60, G)
            output_dir: Çıktı klasörü (None ise proje/output/idealdata)
        """
        self.symbol = symbol
        self.period = period
        self.symbol_short = symbol.replace("VIP'VIP-", "").replace("VIP'", "")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent.parent / "output" / "idealdata"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Oluşturulan dosya isimleri
        self.strategy1_filename: Optional[str] = None
        self.strategy2_filename: Optional[str] = None
        self.robot_filename: Optional[str] = None
    
    def _generate_filename(self, strategy_num: int, vade_tipi: str) -> str:
        """Sistematik dosya adı oluşturur."""
        # Format: S{num}_{sembol}_{periyot}_{vade}_{tarih}
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"S{strategy_num}_{self.symbol_short}_{self.period}DK_{vade_tipi}_{date_str}"
        return filename
    
    def export_strategy1(
        self, 
        params: Dict[str, Any], 
        vade_tipi: str = "ENDEKS"
    ) -> str:
        """
        Strateji 1 (Yatay Filtre + Scoring) kodunu export eder.
        
        Args:
            params: Optimizasyon parametreleri
            vade_tipi: "ENDEKS" veya "SPOT"
            
        Returns:
            Oluşturulan dosya yolu
        """
        filename = self._generate_filename(1, vade_tipi)
        self.strategy1_filename = filename
        
        code = self._generate_strategy1_code(params, vade_tipi)
        
        filepath = self.output_dir / f"{filename}.cs"
        filepath.write_text(code, encoding='utf-8')
        
        # Parametreleri JSON olarak da kaydet
        params_path = self.output_dir / f"{filename}_params.json"
        params_path.write_text(json.dumps(params, indent=2, default=str), encoding='utf-8')
        
        return str(filepath)
    
    def export_strategy2(
        self, 
        params: Dict[str, Any], 
        vade_tipi: str = "ENDEKS"
    ) -> str:
        """
        Strateji 2 (ARS Trend) kodunu export eder.
        
        Args:
            params: Optimizasyon parametreleri
            vade_tipi: "ENDEKS" veya "SPOT"
            
        Returns:
            Oluşturulan dosya yolu
        """
        filename = self._generate_filename(2, vade_tipi)
        self.strategy2_filename = filename
        
        code = self._generate_strategy2_code(params, vade_tipi)
        
        filepath = self.output_dir / f"{filename}.cs"
        filepath.write_text(code, encoding='utf-8')
        
        # Parametreleri JSON olarak da kaydet
        params_path = self.output_dir / f"{filename}_params.json"
        params_path.write_text(json.dumps(params, indent=2, default=str), encoding='utf-8')
        
        return str(filepath)
    
    def export_combined_robot(self, lot_size: int = 1) -> str:
        """
        S1 + S2 birleşik robot kodunu export eder.
        
        Args:
            lot_size: İşlem lot miktarı
            
        Returns:
            Oluşturulan dosya yolu
        """
        if not self.strategy1_filename or not self.strategy2_filename:
            raise ValueError("Önce strategy1 ve strategy2 export edilmeli!")
        
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"Robot_{self.symbol_short}_{self.period}DK_{date_str}"
        self.robot_filename = filename
        
        code = self._generate_robot_code(lot_size)
        
        filepath = self.output_dir / f"{filename}.cs"
        filepath.write_text(code, encoding='utf-8')
        
        return str(filepath)
    
    def _generate_strategy1_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 1 IdealData kodu oluşturur."""
        
        # Varsayılan parametreler
        p = {
            'adx_period': params.get('adx_period', 14),
            'adx_threshold': params.get('adx_threshold', 25),
            'macdv_fast': params.get('macdv_fast', 12),
            'macdv_slow': params.get('macdv_slow', 26),
            'macdv_signal': params.get('macdv_signal', 9),
            'ema_short': params.get('ema_short', 10),
            'ema_long': params.get('ema_long', 50),
            'atr_period': params.get('atr_period', 14),
            'volume_period': params.get('volume_period', 20),
            'long_score_threshold': params.get('long_score_threshold', 3),
            'short_score_threshold': params.get('short_score_threshold', 3),
        }
        
        code = f'''// ===============================================================================================
// STRATEJİ 1: YATAY FİLTRE + SCORING SİSTEMİ
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- PARAMETRELER ---
var P_ADX_Period = {p['adx_period']};
var P_ADX_Threshold = {p['adx_threshold']};
var P_MACDV_Fast = {p['macdv_fast']};
var P_MACDV_Slow = {p['macdv_slow']};
var P_MACDV_Signal = {p['macdv_signal']};
var P_EMA_Short = {p['ema_short']};
var P_EMA_Long = {p['ema_long']};
var P_ATR_Period = {p['atr_period']};
var P_Volume_Period = {p['volume_period']};
var P_Long_Score_Threshold = {p['long_score_threshold']};
var P_Short_Score_Threshold = {p['short_score_threshold']};

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
{{
    if (ATR[i] > 0)
        MACD_Line[i] = (EMA_Fast[i] - EMA_Slow[i]) / ATR[i];
}}
var MACD_Sig_EMA = Sistem.MA(MACD_Line, "Exp", P_MACDV_Signal);
for (int i = 0; i < V.Count; i++)
{{
    MACD_Signal[i] = MACD_Sig_EMA[i];
    MACD_Hist[i] = MACD_Line[i] - MACD_Signal[i];
}}

// --- YATAY FİLTRE ---
var YatayFiltre = Sistem.Liste(0);
for (int i = 0; i < V.Count; i++)
{{
    YatayFiltre[i] = (ADX[i] < P_ADX_Threshold) ? 1 : 0;
}}

// --- SCORING ---
var LongScore = Sistem.Liste(0);
var ShortScore = Sistem.Liste(0);

for (int i = 1; i < V.Count; i++)
{{
    int ls = 0, ss = 0;
    
    // EMA Trend
    if (EMA_Short[i] > EMA_Long[i]) ls++; else ss++;
    
    // MACD-V
    if (MACD_Hist[i] > 0) ls++; else ss++;
    if (MACD_Hist[i] > MACD_Hist[i-1]) ls++; else ss++;
    
    // Volume
    if (Vol[i] > Vol_SMA[i]) {{ ls++; ss++; }}
    
    // Momentum
    if (C[i] > C[i-1]) ls++; else ss++;
    
    LongScore[i] = ls;
    ShortScore[i] = ss;
}}

// --- SİNYAL ÜRETİMİ ---
var SonYon = "";
for (int i = 1; i < V.Count; i++)
{{
    var Sinyal = "";
    
    // Yatay filtre aktif ve skor yeterli
    if (YatayFiltre[i] == 1)
    {{
        if (LongScore[i] >= P_Long_Score_Threshold && SonYon != "A")
            Sinyal = "A";
        else if (ShortScore[i] >= P_Short_Score_Threshold && SonYon != "S")
            Sinyal = "S";
    }}
    else
    {{
        // Yatay değilse flat
        if (SonYon == "A" || SonYon == "S")
            Sinyal = "F";
    }}
    
    if (Sinyal != "" && Sinyal != SonYon)
    {{
        Sistem.Yon[i] = Sinyal;
        SonYon = Sinyal;
    }}
}}

// --- GÖSTERGELERİ ÇİZ ---
Sistem.Cizgiler[0].Deger = ADX;
Sistem.Cizgiler[0].Aciklama = "ADX";
Sistem.Cizgiler[1].Deger = LongScore;
Sistem.Cizgiler[1].Aciklama = "Long Score";
Sistem.Cizgiler[2].Deger = ShortScore;
Sistem.Cizgiler[2].Aciklama = "Short Score";
'''
        return code
    
    def _generate_strategy2_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 2 IdealData kodu oluşturur."""
        
        p = {
            'ars_ema': params.get('ars_ema', 3),
            'ars_atr_p': params.get('ars_atr_p', 14),
            'ars_atr_m': params.get('ars_atr_m', 0.5),
            'momentum_p': params.get('momentum_p', 8),
            'breakout_p1': params.get('breakout_p1', 8),
            'breakout_p2': params.get('breakout_p2', 20),
            'breakout_p3': params.get('breakout_p3', 50),
            'mfi_p': params.get('mfi_p', 14),
            'mfi_hhv_p': params.get('mfi_hhv_p', 14),
            'vol_p': params.get('vol_p', 20),
            'atr_exit_p': params.get('atr_exit_p', 14),
            'atr_sl_mult': params.get('atr_sl_mult', 2.0),
            'atr_tp_mult': params.get('atr_tp_mult', 3.0),
            'atr_trail_mult': params.get('atr_trail_mult', 1.5),
            'exit_confirm_bars': params.get('exit_confirm_bars', 2),
            'exit_confirm_mult': params.get('exit_confirm_mult', 1.0),
        }
        
        code = f'''// ===============================================================================================
// STRATEJİ 2: ARS TREND TAKİP SİSTEMİ
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- ARS PARAMETRELER ---
var P_ARS_EMA = {p['ars_ema']};
var P_ARS_ATR_Period = {p['ars_atr_p']};
var P_ARS_ATR_Mult = {p['ars_atr_m']};

// --- GİRİŞ PARAMETRELER ---
var P_Momentum = {p['momentum_p']};
var P_Breakout_Short = {p['breakout_p1']};
var P_Breakout_Mid = {p['breakout_p2']};
var P_Breakout_Long = {p['breakout_p3']};
var P_MFI_Period = {p['mfi_p']};
var P_MFI_HHV = {p['mfi_hhv_p']};
var P_Volume_Period = {p['vol_p']};

// --- ÇIKIŞ PARAMETRELER ---
var P_ATR_Exit = {p['atr_exit_p']};
var P_ATR_SL_Mult = {p['atr_sl_mult']};
var P_ATR_TP_Mult = {p['atr_tp_mult']};
var P_ATR_Trail_Mult = {p['atr_trail_mult']};
var P_Exit_Confirm_Bars = {p['exit_confirm_bars']};
var P_Exit_Confirm_Mult = {p['exit_confirm_mult']};

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
{{
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
}}

// --- TREND ---
var TrendYonu = Sistem.Liste(0);
for (int i = 1; i < V.Count; i++)
{{
    if (C[i] > ARS[i]) TrendYonu[i] = 1;
    else if (C[i] < ARS[i]) TrendYonu[i] = -1;
    else TrendYonu[i] = TrendYonu[i-1];
}}

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
{{
    var Sinyal = "";
    
    // === ÇIKIŞ MANTIĞI ===
    if (SonYon == "A")
    {{
        // Stop Loss
        if (L[i] <= stopFiyat) Sinyal = "F";
        // Take Profit
        else if (H[i] >= hedefFiyat) Sinyal = "F";
        // Trailing Stop
        else if (C[i] < trailStop) Sinyal = "F";
        // Trend dönüşü (çift teyit)
        else if (TrendYonu[i] == -1)
        {{
            trendDonusCounter++;
            if (trendDonusCounter >= P_Exit_Confirm_Bars) Sinyal = "F";
        }}
        else trendDonusCounter = 0;
        
        // Trail güncelle
        if (Sinyal == "")
        {{
            float yeniTrail = H[i] - ATR_Exit[i] * (float)P_ATR_Trail_Mult;
            if (yeniTrail > trailStop) trailStop = yeniTrail;
        }}
    }}
    else if (SonYon == "S")
    {{
        if (H[i] >= stopFiyat) Sinyal = "F";
        else if (L[i] <= hedefFiyat) Sinyal = "F";
        else if (C[i] > trailStop) Sinyal = "F";
        else if (TrendYonu[i] == 1)
        {{
            trendDonusCounter++;
            if (trendDonusCounter >= P_Exit_Confirm_Bars) Sinyal = "F";
        }}
        else trendDonusCounter = 0;
        
        if (Sinyal == "")
        {{
            float yeniTrail = L[i] + ATR_Exit[i] * (float)P_ATR_Trail_Mult;
            if (yeniTrail < trailStop) trailStop = yeniTrail;
        }}
    }}
    
    // === GİRİŞ MANTIĞI ===
    if (Sinyal == "" && SonYon != "A" && SonYon != "S")
    {{
        bool mfiBreakout = MFI[i] >= MFI_HHV[i-1];
        
        if (TrendYonu[i] == 1)
        {{
            bool yeniZirve = H[i] >= HHV_Short[i-1];
            bool momentum = Momentum[i] > 100;
            
            if (yeniZirve && momentum && mfiBreakout)
            {{
                Sinyal = "A";
                girisFiyat = C[i];
                stopFiyat = girisFiyat - ATR_Exit[i] * (float)P_ATR_SL_Mult;
                hedefFiyat = girisFiyat + ATR_Exit[i] * (float)P_ATR_TP_Mult;
                trailStop = stopFiyat;
                trendDonusCounter = 0;
            }}
        }}
        else if (TrendYonu[i] == -1)
        {{
            bool yeniDip = L[i] <= LLV_Short[i-1];
            bool momentum = Momentum[i] < 100;
            
            if (yeniDip && momentum && mfiBreakout)
            {{
                Sinyal = "S";
                girisFiyat = C[i];
                stopFiyat = girisFiyat + ATR_Exit[i] * (float)P_ATR_SL_Mult;
                hedefFiyat = girisFiyat - ATR_Exit[i] * (float)P_ATR_TP_Mult;
                trailStop = stopFiyat;
                trendDonusCounter = 0;
            }}
        }}
    }}
    
    if (Sinyal != "" && Sinyal != SonYon)
    {{
        Sistem.Yon[i] = Sinyal;
        SonYon = Sinyal;
    }}
}}

// --- GÖSTERGELERİ ÇİZ ---
Sistem.Cizgiler[0].Deger = ARS;
Sistem.Cizgiler[0].Aciklama = "ARS";
Sistem.Cizgiler[1].Deger = HHV_Short;
Sistem.Cizgiler[1].Aciklama = "HHV";
Sistem.Cizgiler[2].Deger = LLV_Short;
Sistem.Cizgiler[2].Aciklama = "LLV";
'''
        return code
    
    def _generate_robot_code(self, lot_size: int) -> str:
        """S1 + S2 birleşik robot kodu oluşturur."""
        
        code = f'''// ===============================================================================================
// ROBOT: S1 + S2 BİRLEŞİK İŞLEM ROBOTU
// ===============================================================================================
// Strateji 1: {self.strategy1_filename} (Gatekeeper - Yön Belirler)
// Strateji 2: {self.strategy2_filename} (İşlem Motoru)
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Lot: {lot_size}
// Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- AYARLAR ---
var LotSize = {lot_size};
var Sembol = "{self.symbol}";
var Periyot = "{self.period}";

// --- STRATEJİLERİ GETİR ---
var Sistem1 = Sistem.SistemGetir("{self.strategy1_filename}", Sembol, Periyot);
var Sistem2 = Sistem.SistemGetir("{self.strategy2_filename}", Sembol, Periyot);

if (Sistem1 == null || Sistem2 == null)
{{
    Sistem.Mesaj(Sistem.Name + " - Strateji bulunamadı!", Color.Red);
    return null;
}}

// --- SAAT KONTROLÜ ---
var Saat = DateTime.Now.TimeOfDay;
bool SeansSaati = (Saat >= new TimeSpan(9, 30, 0) && Saat < new TimeSpan(18, 15, 0)) ||
                  (Saat >= new TimeSpan(19, 0, 0) && Saat < new TimeSpan(23, 0, 0));

if (!SeansSaati)
{{
    return null;
}}

// --- SİNYAL BİRLEŞTİRME ---
// S1: Kapı açık mı? (Yön belirleme)
// S2: İşlem sinyali var mı?

var Yon1 = Sistem1.SonYon;  // A, S veya F
var Yon2 = Sistem2.SonYon;  // A, S veya F

string Sinyal = "";

// Kapı LONG açık ve S2 LONG sinyali
if (Yon1 == "A" && Yon2 == "A")
{{
    Sinyal = "A";
}}
// Kapı SHORT açık ve S2 SHORT sinyali
else if (Yon1 == "S" && Yon2 == "S")
{{
    Sinyal = "S";
}}
// Kapı kapalı veya sinyal uyuşmazlığı
else
{{
    Sinyal = "F";
}}

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
{{
    // Pozisyonu kapat
    Miktar = -Pozisyon;
    Rezerv = "POZİSYON KAPATILDI";
}}
else if (Sinyal == "A" && Pozisyon != LotSize)
{{
    // Long aç/artır
    Miktar = LotSize - Pozisyon;
    Rezerv = "LONG AÇ";
}}
else if (Sinyal == "S" && Pozisyon != -LotSize)
{{
    // Short aç/artır
    Miktar = -LotSize - Pozisyon;
    Rezerv = "SHORT AÇ";
}}

// --- EMİR GÖNDER ---
if (Miktar != 0)
{{
    var Islem = Miktar > 0 ? "ALIS" : "SATIS";
    
    Sistem.PozisyonKontrolGuncelle(Anahtar, Miktar + Pozisyon, SonFiyat, Rezerv);
    
    Sistem.EmirSembol = EmirSembol;
    Sistem.EmirIslem = Islem;
    Sistem.EmirSuresi = "KIE";
    Sistem.EmirTipi = "Piyasa";
    Sistem.EmirMiktari = Math.Abs(Miktar);
    Sistem.EmirGonder();
    
    Sistem.Mesaj(Sistem.Name + " | " + Rezerv + " | Lot: " + Math.Abs(Miktar), Color.Green);
}}

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
'''
        return code
    
    def export_all(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
        vade_tipi: str = "ENDEKS",
        lot_size: int = 1
    ) -> Dict[str, str]:
        """
        Tüm dosyaları tek seferde export eder.
        
        Returns:
            Dict with 'strategy1', 'strategy2', 'robot' paths
        """
        s1_path = self.export_strategy1(params1, vade_tipi)
        s2_path = self.export_strategy2(params2, vade_tipi)
        robot_path = self.export_combined_robot(lot_size)
        
        return {
            'strategy1': s1_path,
            'strategy2': s2_path,
            'robot': robot_path
        }


# --- TEST ---
if __name__ == "__main__":
    # Örnek parametreler
    params1 = {
        'adx_period': 14,
        'adx_threshold': 25,
        'macdv_fast': 12,
        'macdv_slow': 26,
        'long_score_threshold': 3,
    }
    
    params2 = {
        'ars_ema': 3,
        'ars_atr_p': 14,
        'ars_atr_m': 0.5,
        'momentum_p': 8,
        'breakout_p1': 8,
        'mfi_p': 14,
        'atr_sl_mult': 2.0,
        'atr_tp_mult': 3.0,
    }
    
    exporter = IdealDataExporter(
        symbol="VIP'VIP-X030",
        period="5"
    )
    
    result = exporter.export_all(params1, params2, "ENDEKS", 1)
    
    print("Export tamamlandı!")
    print(f"Strateji 1: {result['strategy1']}")
    print(f"Strateji 2: {result['strategy2']}")
    print(f"Robot: {result['robot']}")
