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
        self.strategy3_filename: Optional[str] = None
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
    


    def _get_performance_panel_code(self) -> str:
        """Kullanıcının talep ettiği standart performans paneli kodu."""
        return '''
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
'''

    def _generate_strategy1_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 1 IdealData kodu oluşturur (v4.1 - Bayram/Vade Yönetimi dahil)."""
        
        # Varsayılan parametreler (v4.1) - TÜM parametreler dahil
        # Period parametreleri int() ile sarmalanarak integer olmaları garanti edilir
        p = {
            'min_score': int(params.get('min_score', 3)),
            'exit_score': int(params.get('exit_score', 3)),
            'ars_period': int(params.get('ars_period', 3)),
            'ars_k': params.get('ars_k', 0.01),
            'adx_period': int(params.get('adx_period', 17)),
            'adx_threshold': params.get('adx_threshold', 25.0),
            'netlot_threshold': params.get('netlot_threshold', 20),
            'netlot_period': int(params.get('netlot_period', 5)),
            'macdv_short': int(params.get('macdv_short', 13)),
            'macdv_long': int(params.get('macdv_long', 28)),
            'macdv_signal': int(params.get('macdv_signal', 8)),
            'macdv_threshold': params.get('macdv_threshold', 0),
            # Yatay Filtre parametreleri
            'yatay_ars_bars': int(params.get('yatay_ars_bars', 10)),
            'ars_mesafe_threshold': params.get('ars_mesafe_threshold', 0.25),
            'yatay_adx_threshold': params.get('yatay_adx_threshold', 20.0),
            'bb_period': int(params.get('bb_period', 20)),
            'bb_std': params.get('bb_std', 2.0),
            'bb_width_multiplier': params.get('bb_width_multiplier', 0.8),
            'bb_avg_period': int(params.get('bb_avg_period', 50)),
            'filter_score_threshold': int(params.get('filter_score_threshold', 2)),
        }

        

        code = f'''// ===============================================================================================
// STRATEJI 1: GATEKEEPER (MACDV + ARS + ADX + NETLOT)
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- PARAMETRELER ---
var MIN_ONAY_SKORU = {p['min_score']};
var CIKIS_HASSASIYETI = {p['exit_score']};
var ARS_PERIYOT = {p['ars_period']};
var ARS_K = {p['ars_k']};
var ADX_PERIOD = {p['adx_period']};
var ADX_ESIK = {p['adx_threshold']}f;
var NETLOT_ESIK = {p['netlot_threshold']}f;
var NETLOT_PERIOD = {p['netlot_period']};
var MACDV_K = {p['macdv_short']};
var MACDV_U = {p['macdv_long']};
var MACDV_SIG = {p['macdv_signal']};
var MACDV_ESIK = {p['macdv_threshold']}f;

// --- YATAY FİLTRE PARAMETRELERİ ---
var YATAY_ARS_BARS = {p['yatay_ars_bars']};
var ARS_MESAFE_ESIK = {p['ars_mesafe_threshold']}f;
var YATAY_ADX_ESIK = {p['yatay_adx_threshold']}f;
var BB_PERIOD = {p['bb_period']};
var BB_STD = {p['bb_std']}f;
var BB_WIDTH_MULT = {p['bb_width_multiplier']}f;
var BB_AVG_PERIOD = {p['bb_avg_period']};
var FILTRE_SKOR_ESIK = {p['filter_score_threshold']};

// --- VADE TİPİ ---
string VadeTipi = "{vade_tipi}";


// ===============================================================================================
// DİNAMİK BAYRAM TARİHLERİ (2024-2030)
// ===============================================================================================
int yil = DateTime.Now.Year;
DateTime Ramazan, Kurban;

switch(yil)
{{
    case 2024: Ramazan = new DateTime(2024, 4, 10); Kurban = new DateTime(2024, 6, 16); break;
    case 2025: Ramazan = new DateTime(2025, 3, 30); Kurban = new DateTime(2025, 6, 6); break;
    case 2026: Ramazan = new DateTime(2026, 3, 20); Kurban = new DateTime(2026, 5, 27); break;
    case 2027: Ramazan = new DateTime(2027, 3, 9); Kurban = new DateTime(2027, 5, 16); break;
    case 2028: Ramazan = new DateTime(2028, 2, 26); Kurban = new DateTime(2028, 5, 5); break;
    case 2029: Ramazan = new DateTime(2029, 2, 14); Kurban = new DateTime(2029, 4, 24); break;
    case 2030: Ramazan = new DateTime(2030, 2, 3); Kurban = new DateTime(2030, 4, 13); break;
    default: Ramazan = new DateTime(yil, 3, 15); Kurban = new DateTime(yil, 5, 20); break;
}}

// Optimizasyon için tüm yıllar
DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] {{ "01.01","04.23","05.01","05.19","07.15","08.30","10.29" }};

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
{{
    var aySonu = new DateTime(dt.Year, dt.Month, DateTime.DaysInMonth(dt.Year, dt.Month));
    var d = aySonu;
    
    for (int k = 0; k < 15; k++)
    {{
        if (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
        {{ d = d.AddDays(-1); continue; }}
        
        string mmdd = d.ToString("MM.dd");
        bool tatil = false;
        for (int t = 0; t < resmiTatiller.Length; t++)
            if (resmiTatiller[t] == mmdd) {{ tatil = true; break; }}
        if (tatil) {{ d = d.AddDays(-1); continue; }}
        
        if ((d >= R2024 && d <= R2024.AddDays(3)) || (d >= K2024 && d <= K2024.AddDays(4)) ||
            (d >= R2025 && d <= R2025.AddDays(3)) || (d >= K2025 && d <= K2025.AddDays(4)) ||
            (d >= R2026 && d <= R2026.AddDays(3)) || (d >= K2026 && d <= K2026.AddDays(4)) ||
            (d >= R2027 && d <= R2027.AddDays(3)) || (d >= K2027 && d <= K2027.AddDays(4)))
        {{ d = d.AddDays(-1); continue; }}
        
        break;
    }}
    return d.Date;
}};

// --- 1. ARS ---
var ARS_EMA = Sistem.MA(T, "Exp", ARS_PERIYOT);
var ARS = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {{
    float altBand = (float)(ARS_EMA[i] * (1 - ARS_K));
    float ustBand = (float)(ARS_EMA[i] * (1 + ARS_K));
    if (altBand > ARS[i - 1]) ARS[i] = altBand;
    else if (ustBand < ARS[i - 1]) ARS[i] = ustBand;
    else ARS[i] = ARS[i - 1];
}}

// --- 2. MACDV ---
var EMA_S = Sistem.MA(C, "Exp", MACDV_K);
var EMA_L = Sistem.MA(C, "Exp", MACDV_U);

var TR_List = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {{
    float hl = H[i] - L[i];
    float hc = Math.Abs(H[i] - C[i-1]);
    float lc = Math.Abs(L[i] - C[i-1]);
    TR_List[i] = Math.Max(hl, Math.Max(hc, lc));
}}
var ATRe = Sistem.MA(TR_List, "Exp", MACDV_U);

var MACDV = Sistem.Liste(0);
for (int i = 0; i < Sistem.BarSayisi; i++) {{
    if (ATRe[i] != 0)
        MACDV[i] = ((EMA_S[i] - EMA_L[i]) / ATRe[i]) * 100;
}}
var MACDV_Sinyal = Sistem.MA(MACDV, "Exp", MACDV_SIG);

// --- 3. YATAY FILTRE ---
var ARS_Degisim = Sistem.Liste(0);
for (int i = YATAY_ARS_BARS; i < Sistem.BarSayisi; i++) {{
    bool arsAyni = true;
    for (int j = 1; j <= YATAY_ARS_BARS; j++)
        if (ARS[i] != ARS[i - j]) {{ arsAyni = false; break; }}
    ARS_Degisim[i] = arsAyni ? 0 : 1;
}}

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
for (int i = BB_AVG_PERIOD; i < Sistem.BarSayisi; i++) {{
    int skor = 0;
    if (ARS_Degisim[i] == 1) skor++;
    if (ARS_Mesafe[i] > ARS_MESAFE_ESIK) skor++;
    if (ADX14[i] > YATAY_ADX_ESIK) skor++;
    if (BBWidth[i] > BBWidth_Avg[i] * BB_WIDTH_MULT) skor++;
    YatayFiltre[i] = (skor >= FILTRE_SKOR_ESIK) ? 1 : 0;
}}

// --- 4. NET HACİM ---
var NetLot = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++) {{
    float barHacim = (H[i] - L[i]) > 0 ? (C[i] - O[i]) / (H[i] - L[i]) : 0;
    NetLot[i] = barHacim * 100;
}}
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
{{
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
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }}
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }}
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }}
    
    if (Sinyal == "F")
    {{
        if (SonYon != Sinyal) {{ Sistem.Yon[i] = Sinyal; SonYon = Sinyal; }}
        continue;
    }}
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0)))
        continue;
    
    if (warmupAktif && warmupBaslangicBar == -999)
    {{
        bool yeniSeansBaslangici = false;
        if (aksamSeansi && i > 0 && V[i-1].Date.TimeOfDay < new TimeSpan(19,0,0))
            yeniSeansBaslangici = true;
        if (gunSeansi && t >= new TimeSpan(9,30,0) && t < new TimeSpan(9,35,0))
            if (i > 0 && dt.Date != V[i-1].Date.Date)
                yeniSeansBaslangici = true;
        if (yeniSeansBaslangici)
            warmupBaslangicBar = i;
    }}
    
    if (warmupAktif && warmupBaslangicBar > 0)
    {{
        if ((i - warmupBaslangicBar) < vadeCooldownBar) continue;
        else warmupAktif = false;
    }}
    
    if (arefeFlat && i > 0 && dt.Date != V[i-1].Date.Date)
        arefeFlat = false;
    
    // --- SKORLAMA ---
    int longScore = 0;
    int shortScore = 0;

    if (C[i] > ARS[i]) longScore++; else if (C[i] < ARS[i]) shortScore++;
    if (MACDV[i] > (MACDV_Sinyal[i] + MACDV_ESIK)) longScore++; else if (MACDV[i] < (MACDV_Sinyal[i] - MACDV_ESIK)) shortScore++;
    if (NetLot_MA[i] > NETLOT_ESIK) longScore++; else if (NetLot_MA[i] < -NETLOT_ESIK) shortScore++;
    if (ADX14[i] > ADX_ESIK) {{ longScore++; shortScore++; }}


    // --- ÇIKIŞ MANTIĞI ---
    if (SonYon == "A") {{
        if (C[i] < ARS[i] || shortScore >= CIKIS_HASSASIYETI) Sinyal = "F";
    }}
    else if (SonYon == "S") {{
        if (C[i] > ARS[i] || longScore >= CIKIS_HASSASIYETI) Sinyal = "F";
    }}
    
    // --- GİRİŞ MANTIĞI ---
    if (Sinyal == "" && SonYon != "A" && SonYon != "S") {{
        if (YatayFiltre[i] == 1) {{
            if (longScore >= MIN_ONAY_SKORU && shortScore < 2) Sinyal = "A";
            else if (shortScore >= MIN_ONAY_SKORU && longScore < 2) Sinyal = "S";
        }}
    }}
    
    // --- POZİSYON GÜNCELLEME ---
    if (Sinyal != "" && SonYon != Sinyal) {{
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
    }}
}}

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

{self._get_performance_panel_code()}
'''
        return code

    
    def _generate_strategy2_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 2 IdealData kodu oluşturur (v4.1 - Bayram/Vade/Volume/DC Exit dahil)."""
        
        p = {
            'ars_ema': int(params.get('ars_ema_period', 3)),
            'ars_atr_p': int(params.get('ars_atr_period', 10)),
            'ars_atr_m': params.get('ars_atr_mult', 0.5),
            'ars_min_band': params.get('ars_min_band', 0.002),
            'ars_max_band': params.get('ars_max_band', 0.015),
            'momentum_p': int(params.get('momentum_period', 5)),
            'momentum_threshold': params.get('momentum_threshold', 100.0),
            'momentum_base': params.get('momentum_base', 200.0),
            'breakout_p': int(params.get('breakout_period', 10)),
            'mfi_p': int(params.get('mfi_period', 14)),
            'mfi_hhv_p': int(params.get('mfi_hhv_period', 14)),
            'mfi_llv_p': int(params.get('mfi_llv_period', 14)),
            'volume_hhv_p': int(params.get('volume_hhv_period', 14)),
            'atr_exit_p': int(params.get('atr_exit_period', 14)),
            'atr_sl_mult': params.get('atr_sl_mult', 2.0),
            'atr_tp_mult': params.get('atr_tp_mult', 5.0),
            'atr_trail_mult': params.get('atr_trail_mult', 2.0),
            'exit_confirm_bars': int(params.get('exit_confirm_bars', 2)),
            'exit_confirm_mult': params.get('exit_confirm_mult', 1.0),
            'volume_mult': params.get('volume_mult', 0.8),
        }
        
        code = f'''// ===============================================================================================
// STRATEJİ 2: ARS TREND TAKİP SİSTEMİ v4.1
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- VADE TİPİ ---
string VadeTipi = "{vade_tipi}";

// --- ATR EXIT PARAMETRELER ---
int ATR_Exit_Period = {p['atr_exit_p']};
double ATR_SL_Mult = {p['atr_sl_mult']};
double ATR_TP_Mult = {p['atr_tp_mult']};
double ATR_Trail_Mult = {p['atr_trail_mult']};
int Exit_Confirm_Bars = {p['exit_confirm_bars']};
double Exit_Confirm_Mult = {p['exit_confirm_mult']};

// --- ARS PARAMETRELER ---
int ARS_EMA_Period = {p['ars_ema']};
int ARS_ATR_Period = {p['ars_atr_p']};
double ARS_ATR_Mult = {p['ars_atr_m']};
double ARS_Min_Band = {p['ars_min_band']};
double ARS_Max_Band = {p['ars_max_band']};

// --- GİRİŞ SİNYALİ PARAMETRELER ---
int MOMENTUM_Period = {p['momentum_p']};
double MOMENTUM_THRESHOLD = {p['momentum_threshold']};
double MOMENTUM_BASE = {p['momentum_base']};
int BREAKOUT_Period = {p['breakout_p']};
double VOLUME_MULT = {p['volume_mult']};

// ===============================================================================================
// DİNAMİK BAYRAM TARİHLERİ (2024-2030)
// ===============================================================================================
int yil = DateTime.Now.Year;
DateTime Ramazan, Kurban;

switch(yil)
{{
    case 2024: Ramazan = new DateTime(2024, 4, 10); Kurban = new DateTime(2024, 6, 16); break;
    case 2025: Ramazan = new DateTime(2025, 3, 30); Kurban = new DateTime(2025, 6, 6); break;
    case 2026: Ramazan = new DateTime(2026, 3, 20); Kurban = new DateTime(2026, 5, 27); break;
    case 2027: Ramazan = new DateTime(2027, 3, 9); Kurban = new DateTime(2027, 5, 16); break;
    case 2028: Ramazan = new DateTime(2028, 2, 26); Kurban = new DateTime(2028, 5, 5); break;
    case 2029: Ramazan = new DateTime(2029, 2, 14); Kurban = new DateTime(2029, 4, 24); break;
    case 2030: Ramazan = new DateTime(2030, 2, 3); Kurban = new DateTime(2030, 4, 13); break;
    default: Ramazan = new DateTime(yil, 3, 15); Kurban = new DateTime(yil, 5, 20); break;
}}

DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] {{ "01.01","04.23","05.01","05.19","07.15","08.30","10.29" }};

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
{{
    float dinamikK;
    if (ARS_ATR_Mult > 0) {{
        dinamikK = (ATR[i] / ARS_EMA[i]) * (float)ARS_ATR_Mult;
        dinamikK = Math.Max((float)ARS_Min_Band, Math.Min((float)ARS_Max_Band, dinamikK));
    }} else {{
        dinamikK = (float)ARS_Min_Band;
    }}
    
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
}}

// TREND BELİRLEME
var TrendYonu = Sistem.Liste(0);
for (int i = 1; i < Sistem.BarSayisi; i++)
{{
    if (C[i] > ARS[i]) TrendYonu[i] = 1;
    else if (C[i] < ARS[i]) TrendYonu[i] = -1;
    else TrendYonu[i] = TrendYonu[i-1];
}}

var ATR_Exit = Sistem.AverageTrueRange(ATR_Exit_Period);

// GİRİŞ SİNYAL İNDİKATÖRLERİ
var Momentum = Sistem.Momentum(MOMENTUM_Period);
var HHV = Sistem.HHV(BREAKOUT_Period, H);
var LLV = Sistem.LLV(BREAKOUT_Period, L);

var MFI = Sistem.MoneyFlowIndex({p['mfi_p']});
var MFI_HHV = Sistem.HHV({p['mfi_hhv_p']}, MFI);
var MFI_LLV = Sistem.LLV({p['mfi_llv_p']}, MFI);

var Vol_HHV = Sistem.HHV({p['volume_hhv_p']}, Lot);

// ===============================================================================================
// VADE SONU İŞ GÜNÜ HESAPLAMA
// ===============================================================================================
Func<DateTime, DateTime> VadeSonuIsGunu = (dt) =>
{{
    var aySonu = new DateTime(dt.Year, dt.Month, DateTime.DaysInMonth(dt.Year, dt.Month));
    var d = aySonu;
    
    for (int k = 0; k < 15; k++)
    {{
        if (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
        {{ d = d.AddDays(-1); continue; }}
        
        string mmdd = d.ToString("MM.dd");
        bool tatil = false;
        for (int t = 0; t < resmiTatiller.Length; t++)
            if (resmiTatiller[t] == mmdd) {{ tatil = true; break; }}
        if (tatil) {{ d = d.AddDays(-1); continue; }}
        
        if ((d >= R2024 && d <= R2024.AddDays(3)) || (d >= K2024 && d <= K2024.AddDays(4)) ||
            (d >= R2025 && d <= R2025.AddDays(3)) || (d >= K2025 && d <= K2025.AddDays(4)) ||
            (d >= R2026 && d <= R2026.AddDays(3)) || (d >= K2026 && d <= K2026.AddDays(4)) ||
            (d >= R2027 && d <= R2027.AddDays(3)) || (d >= K2027 && d <= K2027.AddDays(4)))
        {{ d = d.AddDays(-1); continue; }}
        
        break;
    }}
    return d.Date;
}};

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
{{
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
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }}
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }}
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true;
        warmupBaslangicBar = -999;
        arefeFlat = false;
    }}
    
    if (Sinyal == "F")
    {{
        if (SonYon != Sinyal) {{ Sistem.Yon[i] = Sinyal; SonYon = Sinyal; }}
        continue;
    }}
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0)))
        continue;
    
    if (warmupAktif && warmupBaslangicBar == -999)
    {{
        bool yeniSeansBaslangici = false;
        if (aksamSeansi && i > 0 && V[i-1].Date.TimeOfDay < new TimeSpan(19,0,0))
            yeniSeansBaslangici = true;
        if (gunSeansi && t >= new TimeSpan(9,30,0) && t < new TimeSpan(9,35,0))
            if (i > 0 && dt.Date != V[i-1].Date.Date)
                yeniSeansBaslangici = true;
        if (yeniSeansBaslangici)
            warmupBaslangicBar = i;
    }}
    
    if (warmupAktif && warmupBaslangicBar > 0)
    {{
        if ((i - warmupBaslangicBar) < vadeCooldownBar) continue;
        else warmupAktif = false;
    }}
    
    if (arefeFlat && i > 0 && dt.Date != V[i-1].Date.Date)
        arefeFlat = false;
    
    // === ÇIKIŞ MANTIĞI (ATR-Based + Double Confirmation) ===
    if (SonYon == "A")
    {{
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
    }}
    else if (SonYon == "S")
    {{
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
    }}
    
    // === GİRİŞ MANTIĞI ===
    if (Sinyal == "" && SonYon != "A" && SonYon != "S")
    {{
        if (TrendYonu[i] == 1)
        {{
            bool yeniZirve = H[i] >= HHV[i-1] && HHV[i] > HHV[i-1];
            bool pozitifMomentum = Momentum[i] > MOMENTUM_THRESHOLD;
            bool mfiOnay = MFI[i] >= MFI_HHV[i-1];
            bool volumeOnay = Lot[i] >= Vol_HHV[i-1] * (float)VOLUME_MULT;
            if (yeniZirve && pozitifMomentum && mfiOnay && volumeOnay) Sinyal = "A";
        }}
        else if (TrendYonu[i] == -1)
        {{
            bool yeniDip = L[i] <= LLV[i-1] && LLV[i] < LLV[i-1];
            bool negatifMomentum = Momentum[i] < (MOMENTUM_BASE - MOMENTUM_THRESHOLD);
            bool mfiOnay = MFI[i] <= MFI_LLV[i-1];
            bool volumeOnay = Lot[i] >= Vol_HHV[i-1] * (float)VOLUME_MULT;
            if (yeniDip && negatifMomentum && mfiOnay && volumeOnay) Sinyal = "S";
        }}
    }}
    
    if (Sinyal != "" && SonYon != Sinyal)
    {{
        if (Sinyal == "A")
        {{
            entryPrice = C[i];
            entryBar = i;
            extremePrice = H[i];
            belowArsCount = 0;
        }}
        else if (Sinyal == "S")
        {{
            entryPrice = C[i];
            entryBar = i;
            extremePrice = L[i];
            aboveArsCount = 0;
        }}
        else if (Sinyal == "F")
        {{
            entryPrice = 0;
            extremePrice = 0;
            belowArsCount = 0;
            aboveArsCount = 0;
        }}
        
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
    }}
}}

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
{self._get_performance_panel_code()}
'''
        return code

    def _generate_strategy3_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 3 (Paradise) IdealData kodu oluşturur."""
        
        # Parametreler
        p = {
            'ema_period': int(params.get('ema_period', 21)),
            'dsma_period': int(params.get('dsma_period', 50)),
            'ma_period': int(params.get('ma_period', 20)),
            'hh_period': int(params.get('hh_period', 25)),
            'vol_hhv_period': int(params.get('vol_hhv_period', 14)),
            'mom_period': int(params.get('mom_period', 60)),
            'mom_alt': params.get('mom_alt', 98.0),
            'mom_ust': params.get('mom_ust', 102.0),
            'atr_period': int(params.get('atr_period', 14)),
            'atr_sl': params.get('atr_sl', 2.0),
            'atr_tp': params.get('atr_tp', 4.0),
            'atr_trail': params.get('atr_trail', 2.5),
            'yon_modu': params.get('yon_modu', 'CIFT'),
        }

        code = f'''// ===============================================================================================
// STRATEJI 3: PARADISE v2.0 (Vade/Tatil Korumalı)
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Olusturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- VADE TİPİ ---
string VadeTipi = "{vade_tipi}";
string YON_MODU = "{p['yon_modu']}"; // CIFT veya SADECE_AL

// --- PARAMETRELER ---
var ema_period = {p['ema_period']};
var dsma_period = {p['dsma_period']};
var ma_period = {p['ma_period']};
var hh_period = {p['hh_period']};
var vol_hhv_period = {p['vol_hhv_period']};
var mom_period = {p['mom_period']};
var mom_alt = {p['mom_alt']}f;
var mom_ust = {p['mom_ust']}f;
var atr_period = {p['atr_period']};
var atr_sl = {p['atr_sl']}f;
var atr_tp = {p['atr_tp']}f;
var atr_trail = {p['atr_trail']}f;

// ===============================================================================================
// DİNAMİK BAYRAM TARİHLERİ (2024-2030)
// ===============================================================================================
int yil = DateTime.Now.Year;
DateTime Ramazan, Kurban;

switch(yil)
{{
    case 2024: Ramazan = new DateTime(2024, 4, 10); Kurban = new DateTime(2024, 6, 16); break;
    case 2025: Ramazan = new DateTime(2025, 3, 30); Kurban = new DateTime(2025, 6, 6); break;
    case 2026: Ramazan = new DateTime(2026, 3, 20); Kurban = new DateTime(2026, 5, 27); break;
    case 2027: Ramazan = new DateTime(2027, 3, 9); Kurban = new DateTime(2027, 5, 16); break;
    case 2028: Ramazan = new DateTime(2028, 2, 26); Kurban = new DateTime(2028, 5, 5); break;
    case 2029: Ramazan = new DateTime(2029, 2, 14); Kurban = new DateTime(2029, 4, 24); break;
    case 2030: Ramazan = new DateTime(2030, 2, 3); Kurban = new DateTime(2030, 4, 13); break;
    default: Ramazan = new DateTime(yil, 3, 15); Kurban = new DateTime(yil, 5, 20); break;
}}

DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] {{ "01.01","04.23","05.01","05.19","07.15","08.30","10.29" }};

var Veriler = Sistem.GrafikVerileri;
var C = Sistem.GrafikFiyatSec("Kapanis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");
var O = Sistem.GrafikFiyatSec("Acilis");
var V = Sistem.GrafikFiyatSec("Lot");

// ===============================================================================================
// VADE SONU İŞ GÜNÜ HESAPLAMA
// ===============================================================================================
Func<DateTime, DateTime> VadeSonuIsGunu = (dt) =>
{{
    var aySonu = new DateTime(dt.Year, dt.Month, DateTime.DaysInMonth(dt.Year, dt.Month));
    var d = aySonu;
    
    for (int k = 0; k < 15; k++)
    {{
        if (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
        {{ d = d.AddDays(-1); continue; }}
        
        string mmdd = d.ToString("MM.dd");
        bool tatil = false;
        for (int t = 0; t < resmiTatiller.Length; t++)
            if (resmiTatiller[t] == mmdd) {{ tatil = true; break; }}
        if (tatil) {{ d = d.AddDays(-1); continue; }}
        
        if ((d >= R2024 && d <= R2024.AddDays(3)) || (d >= K2024 && d <= K2024.AddDays(4)) ||
            (d >= R2025 && d <= R2025.AddDays(3)) || (d >= K2025 && d <= K2025.AddDays(4)) ||
            (d >= R2026 && d <= R2026.AddDays(3)) || (d >= K2026 && d <= K2026.AddDays(4)) ||
            (d >= R2027 && d <= R2027.AddDays(3)) || (d >= K2027 && d <= K2027.AddDays(4)))
        {{ d = d.AddDays(-1); continue; }}
        
        break;
    }}
    return d.Date;
}};

// --- INDIKATOR HESAPLAMALARI ---
var EMA = Sistem.MA(C, "Exp", ema_period);
var DSMA1 = Sistem.MA(C, "Simple", dsma_period);
var DSMA = Sistem.MA(DSMA1, "Simple", dsma_period);
var MA = Sistem.MA(C, "Simple", ma_period);
var MOM = Sistem.Momentum(mom_period);
var ATR = Sistem.AverageTrueRange(atr_period);

var HH = Sistem.HHV(hh_period, "Yuksek");
var LL = Sistem.LLV(hh_period, "Dusuk");
var VOL_HHV = Sistem.HHV(vol_hhv_period, "Lot");

// --- LOOP & SINYAL ---
var SonYon = "";
var Pos = 0; 
var EntryPrice = 0.0f;
var ExtremePrice = 0.0f;
var EntryATR = 0.0f;

for (int i = 1; i < Veriler.Count; i++) Sistem.Yon[i] = "";

int vadeCooldownBar = Math.Max(dsma_period * 2, Math.Max(hh_period, mom_period)) + 10;
int warmupBars = Math.Max(50, vadeCooldownBar);
int warmupBaslangicBar = -999;
bool warmupAktif = false;
bool arefeFlat = false;

for (int i = warmupBars; i < Veriler.Count; i++)
{{
    // --- VADE VE TATİL KONTROLLERİ ---
    string Sinyal = "";
    var dt = Veriler[i].Date;
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

    // Vade/Arefe Flat Kuralları
    if (arefe && vadeSonuGun && t > new TimeSpan(11,30,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }}
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }}
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }}
    
    if (Sinyal == "F") {{
        if (SonYon != Sinyal) {{ Sistem.Yon[i] = Sinyal; SonYon = Sinyal; Pos = 0; }}
        continue;
    }}
    
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0))) continue;

    // Warmup Kontrolü
    if (warmupAktif && warmupBaslangicBar == -999) {{
        bool yeniSeans = false;
        if (aksamSeansi && i>0 && Veriler[i-1].Date.TimeOfDay < new TimeSpan(19,0,0)) yeniSeans = true;
        if (gunSeansi && i>0 && dt.Date != Veriler[i-1].Date.Date) yeniSeans = true;
        if (yeniSeans) warmupBaslangicBar = i;
    }}
    if (warmupAktif && warmupBaslangicBar > 0) {{
        if ((i - warmupBaslangicBar) < vadeCooldownBar) continue;
        else warmupAktif = false;
    }}
    if (arefeFlat && i>0 && dt.Date != Veriler[i-1].Date.Date) arefeFlat = false;


    // --- ÇIKIŞ KONTROLLERİ (ATR) ---
    if (Pos == 1)
    {{
        if (C[i] > ExtremePrice) ExtremePrice = C[i];
        
        bool exit = false;
        if (C[i] <= EntryPrice - EntryATR * atr_sl) exit = true; 
        else if (C[i] >= EntryPrice + EntryATR * atr_tp) exit = true; 
        else if (C[i] <= ExtremePrice - EntryATR * atr_trail) exit = true; 
        
        if (exit) Sinyal = "F";
    }}
    else if (Pos == -1)
    {{
        if (C[i] < ExtremePrice) ExtremePrice = C[i];
        
        bool exit = false;
        if (C[i] >= EntryPrice + EntryATR * atr_sl) exit = true; 
        else if (C[i] <= EntryPrice - EntryATR * atr_tp) exit = true; 
        else if (C[i] >= ExtremePrice + EntryATR * atr_trail) exit = true; 
        
        if (exit) Sinyal = "F";
    }}

    // --- GİRİŞ KONTROLLERİ ---
    if (Sinyal == "" && Pos == 0)
    {{
        // Momentum Filtresi: Alt ve Üst bant arasında (sıkışma)
        bool mom_bandinda = MOM[i] > mom_alt && MOM[i] < mom_ust;
        bool vol_ok = V[i] >= VOL_HHV[i-1] * 0.8f;
        
        if (mom_bandinda && vol_ok)
        {{
            // LONG: HH Breakout + EMA > DSMA + C > MA + MOM > 100
            bool hh_ok = H[i] > HH[i-1];
            bool trend_ok = EMA[i] > DSMA[i] && C[i] > MA[i];
            bool mom_ok = MOM[i] > 100; // Bant içinde ama 100 üstü
            
            if (hh_ok && trend_ok && mom_ok)
            {{
                Sinyal = "A";
            }}
            // SHORT: LL Breakdown + EMA < DSMA + C < MA + MOM < 100
            else if (YON_MODU == "CIFT")
            {{
                bool ll_ok = L[i] < LL[i-1];
                bool trend_short = EMA[i] < DSMA[i] && C[i] < MA[i];
                bool mom_short = MOM[i] < 100; // Bant içinde ama 100 altı
                
                if (ll_ok && trend_short && mom_short)
                {{
                    Sinyal = "S";
                }}
            }}
        }}
    }}
    
    // --- POZİSYON GÜNCELLEME ---
    if (Sinyal != "" && SonYon != Sinyal)
    {{
        Sistem.Yon[i] = Sinyal;
        SonYon = Sinyal;
        
        if (Sinyal == "A") {{ Pos = 1; EntryPrice = C[i]; ExtremePrice = C[i]; EntryATR = ATR[i]; }}
        else if (Sinyal == "S") {{ Pos = -1; EntryPrice = C[i]; ExtremePrice = C[i]; EntryATR = ATR[i]; }}
        else if (Sinyal == "F") {{ Pos = 0; }}
    }}
}}

Sistem.Cizgiler[0].Deger = EMA;
Sistem.Cizgiler[1].Deger = DSMA;
Sistem.Cizgiler[2].Deger = MA;

{self._get_performance_panel_code()}
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
    
    
    def _generate_strategy4_code(self, params: Dict[str, Any], vade_tipi: str) -> str:
        """Strateji 4 (TOMA + Momentum + TRIX) IdealData kodu oluşturur."""
        
        # Parametreler
        p = {
            'mom_period': int(params.get('mom_period', 1900)),
            'mom_upper': params.get('mom_limit_high', params.get('mom_upper', 101.5)),
            'mom_lower': params.get('mom_limit_low', params.get('mom_lower', 98.0)),
            'trix_period': int(params.get('trix_period', 120)),
            'trix_lb1': int(params.get('trix_lb1', 145)),
            'trix_lb2': int(params.get('trix_lb2', 160)),
            'hhv1_p': int(params.get('hhv1_period', params.get('hh_ll_period', 20))),
            'llv1_p': int(params.get('llv1_period', params.get('hh_ll_period', 20))),
            'hhv2_p': int(params.get('hhv2_period', params.get('hh_ll_long1', 150))),
            'llv2_p': int(params.get('llv2_period', params.get('hh_ll_long2', 190))),
            'hhv3_p': int(params.get('hhv3_period', 150)),
            'llv3_p': int(params.get('llv3_period', 190)),
            'toma_period': int(params.get('toma_period', 2)),
            'toma_opt': params.get('toma_opt', 2.1),
            'kar_al': params.get('kar_al', 0.0),
            'iz_stop': params.get('iz_stop', 0.0),
        }

        code = f'''// ===============================================================================================
// STRATEJI 4: TOMA + MOMENTUM (Karma Sistem)
// ===============================================================================================
// Sembol: {self.symbol}
// Periyot: {self.period} dakika
// Vade Tipi: {vade_tipi}
// Olusturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}
// ===============================================================================================

// --- VADE TİPİ ---
string VadeTipi = "{vade_tipi}";

// --- PARAMETRELER ---
var MOM_PERIOD = {p['mom_period']};
var MOM_UPPER = {p['mom_upper']}f;
var MOM_LOWER = {p['mom_lower']}f;
var TRIX_PERIOD = {p['trix_period']};
var TRIX_LB1 = {p['trix_lb1']};
var TRIX_LB2 = {p['trix_lb2']};
var HHV1_PERIOD = {p['hhv1_p']};
var LLV1_PERIOD = {p['llv1_p']};
var HHV2_PERIOD = {p['hhv2_p']};
var LLV2_PERIOD = {p['llv2_p']};
var HHV3_PERIOD = {p['hhv3_p']};
var LLV3_PERIOD = {p['llv3_p']};
var TOMA_PERIOD = {p['toma_period']};
var TOMA_OPT = {p['toma_opt']}f;
var KAR_AL_YUZDE = {p['kar_al']}f;
var IZLEYEN_STOP_YUZDE = {p['iz_stop']}f;

// ===============================================================================================
// DİNAMİK BAYRAM TARİHLERİ (2024-2030)
// ===============================================================================================
int yil = DateTime.Now.Year;
DateTime Ramazan, Kurban;

switch(yil)
{{
    case 2024: Ramazan = new DateTime(2024, 4, 10); Kurban = new DateTime(2024, 6, 16); break;
    case 2025: Ramazan = new DateTime(2025, 3, 30); Kurban = new DateTime(2025, 6, 6); break;
    case 2026: Ramazan = new DateTime(2026, 3, 20); Kurban = new DateTime(2026, 5, 27); break;
    case 2027: Ramazan = new DateTime(2027, 3, 9); Kurban = new DateTime(2027, 5, 16); break;
    case 2028: Ramazan = new DateTime(2028, 2, 26); Kurban = new DateTime(2028, 5, 5); break;
    case 2029: Ramazan = new DateTime(2029, 2, 14); Kurban = new DateTime(2029, 4, 24); break;
    case 2030: Ramazan = new DateTime(2030, 2, 3); Kurban = new DateTime(2030, 4, 13); break;
    default: Ramazan = new DateTime(yil, 3, 15); Kurban = new DateTime(yil, 5, 20); break;
}}

DateTime R2024 = new DateTime(2024, 4, 10); DateTime K2024 = new DateTime(2024, 6, 16);
DateTime R2025 = new DateTime(2025, 3, 30); DateTime K2025 = new DateTime(2025, 6, 6);
DateTime R2026 = new DateTime(2026, 3, 20); DateTime K2026 = new DateTime(2026, 5, 27);
DateTime R2027 = new DateTime(2027, 3, 9); DateTime K2027 = new DateTime(2027, 5, 16);

string[] resmiTatiller = new string[] {{ "01.01","04.23","05.01","05.19","07.15","08.30","10.29" }};

var V = Sistem.GrafikVerileri;
var O = Sistem.GrafikFiyatSec("Acilis");
var C = Sistem.GrafikFiyatSec("Kapanis");
var H = Sistem.GrafikFiyatSec("Yuksek");
var L = Sistem.GrafikFiyatSec("Dusuk");

// ===============================================================================================
// VADE SONU İŞ GÜNÜ HESAPLAMA
// ===============================================================================================
Func<DateTime, DateTime> VadeSonuIsGunu = (dt) =>
{{
    var aySonu = new DateTime(dt.Year, dt.Month, DateTime.DaysInMonth(dt.Year, dt.Month));
    var d = aySonu;
    
    for (int k = 0; k < 15; k++)
    {{
        if (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday)
        {{ d = d.AddDays(-1); continue; }}
        
        string mmdd = d.ToString("MM.dd");
        bool tatil = false;
        for (int t = 0; t < resmiTatiller.Length; t++)
            if (resmiTatiller[t] == mmdd) {{ tatil = true; break; }}
        if (tatil) {{ d = d.AddDays(-1); continue; }}
        
        if ((d >= R2024 && d <= R2024.AddDays(3)) || (d >= K2024 && d <= K2024.AddDays(4)) ||
            (d >= R2025 && d <= R2025.AddDays(3)) || (d >= K2025 && d <= K2025.AddDays(4)) ||
            (d >= R2026 && d <= R2026.AddDays(3)) || (d >= K2026 && d <= K2026.AddDays(4)) ||
            (d >= R2027 && d <= R2027.AddDays(3)) || (d >= K2027 && d <= K2027.AddDays(4)))
        {{ d = d.AddDays(-1); continue; }}
        
        break;
    }}
    return d.Date;
}};

// --- INDIKATORLER ---
var TOMA_Line = Sistem.TOMA(TOMA_PERIOD, TOMA_OPT);

var HH1 = Sistem.HHV(HHV1_PERIOD, "Yuksek");
var LL1 = Sistem.LLV(LLV1_PERIOD, "Dusuk");

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
double EntryPrice = 0.0;
double ExtremePrice = 0.0;
var Pos = 0;

for (int i = 1; i < V.Count; i++) Sistem.Yon[i] = "";

int warm1 = Math.Max(MOM_PERIOD, TRIX_PERIOD + Math.Max(TRIX_LB1, TRIX_LB2));
int warm2 = Math.Max(HHV1_PERIOD, Math.Max(HHV2_PERIOD, Math.Max(LLV2_PERIOD, Math.Max(HHV3_PERIOD, LLV3_PERIOD))));
int warmupBars = Math.Max(200, Math.Max(warm1, warm2)) + 10;
int warmupBaslangicBar = -999;
bool warmupAktif = false;
bool arefeFlat = false;

for (int i = warmupBars; i < V.Count; i++)
{{
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
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }}
    else if (arefe && !vadeSonuGun && t > new TimeSpan(11,30,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        arefeFlat = true;
    }}
    else if (vadeSonuGun && t > new TimeSpan(17,40,0))
    {{
        if (SonYon != "F") Sinyal = "F";
        warmupAktif = true; warmupBaslangicBar = -999; arefeFlat = false;
    }}
    
    if (Sinyal == "F") {{
        if (SonYon != Sinyal) {{ Sistem.Yon[i] = Sinyal; SonYon = Sinyal; Pos = 0; }}
        continue;
    }}
    
    if ((arefe && t > new TimeSpan(11,30,0)) || (vadeSonuGun && !arefe && t > new TimeSpan(17,40,0))) continue;

    if (warmupAktif && warmupBaslangicBar == -999) {{
        bool yeniSeans = false;
        if (aksamSeansi && i>0 && V[i-1].Date.TimeOfDay < new TimeSpan(19,0,0)) yeniSeans = true;
        if (gunSeansi && i>0 && dt.Date != V[i-1].Date.Date) yeniSeans = true;
        if (yeniSeans) warmupBaslangicBar = i;
    }}
    if (warmupAktif && warmupBaslangicBar > 0) {{
        if ((i - warmupBaslangicBar) < 100) continue; // Min 100 bar cooldown
        else warmupAktif = false;
    }}
    if (arefeFlat && i>0 && dt.Date != V[i-1].Date.Date) arefeFlat = false;


    // --- STRATEJİ MANTIĞI ---
    
    // Kural 1: MOM > ÜST SINIR
    if (MOM1[i] > MOM_UPPER)
    {{
        if (HH2[i] > HH2[i-1] && TRIX1[i] < TRIX1[i-TRIX_LB1] && TRIX1[i] > TRIX1[i-1]) Sinyal = "A"; 
        if (LL2[i] < LL2[i-1] && TRIX1[i] > TRIX1[i-TRIX_LB1] && TRIX1[i] < TRIX1[i-1]) Sinyal = "S"; 
    }}
    
    // Kural 2: MOM < ALT SINIR
    if (MOM1[i] < MOM_LOWER)
    {{
        if (HH3[i] > HH3[i-1] && TRIX2[i] < TRIX2[i-TRIX_LB2] && TRIX2[i] > TRIX2[i-1]) Sinyal = "A"; 
        if (LL3[i] < LL3[i-1] && TRIX2[i] > TRIX2[i-TRIX_LB2] && TRIX2[i] < TRIX2[i-1]) Sinyal = "S"; 
    }}
    
    // Kural 3: TOMA + HHV/LLV (Ana Trend - Öncelikli, önceki sinyalleri ezer)
    if (HH1[i] > HH1[i-1] && C[i] > TOMA_Line[i]) Sinyal = "A";
    if (LL1[i] < LL1[i-1] && C[i] < TOMA_Line[i]) Sinyal = "S";

    // --- POZİSYON GÜNCELLEME (Giriş / Reverse) ---
    if (Sinyal != "" && SonYon != Sinyal)
    {{
        SonYon = Sinyal;
        Sistem.Yon[i] = SonYon;
        EntryPrice = C[i];
        ExtremePrice = C[i];
        if (Sinyal == "A") Pos = 1;
        else if (Sinyal == "S") Pos = -1;
        else Pos = 0;
    }}

    // --- EXIT LOGIC (Kar Al / İzleyen Stop) ---
    if (Pos == 1) {{
        if (ExtremePrice < C[i]) ExtremePrice = C[i];
        if (KAR_AL_YUZDE > 0 && C[i] >= EntryPrice * (1 + KAR_AL_YUZDE/100.0)) {{
            Sistem.Yon[i] = "F"; Pos = 0;
        }}
        if (IZLEYEN_STOP_YUZDE > 0 && C[i] <= ExtremePrice * (1 - IZLEYEN_STOP_YUZDE/100.0)) {{
            Sistem.Yon[i] = "F"; Pos = 0;
        }}
    }}
    else if (Pos == -1) {{
        if (ExtremePrice == 0 || ExtremePrice > C[i]) ExtremePrice = C[i];
        if (KAR_AL_YUZDE > 0 && C[i] <= EntryPrice * (1 - KAR_AL_YUZDE/100.0)) {{
            Sistem.Yon[i] = "F"; Pos = 0;
        }}
        if (IZLEYEN_STOP_YUZDE > 0 && C[i] >= ExtremePrice * (1 + IZLEYEN_STOP_YUZDE/100.0)) {{
            Sistem.Yon[i] = "F"; Pos = 0;
        }}
    }}
}}

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

{self._get_performance_panel_code()}
'''
        return code

    def export_strategy4(
        self, 
        params: Dict[str, Any], 
        vade_tipi: str = "ENDEKS"
    ) -> str:
        """
        Strateji 4 Kodunu Export Eder
        """
        filename = self._generate_filename(4, vade_tipi)
        
        code = self._generate_strategy4_code(params, vade_tipi)
        
        filepath = self.output_dir / f"{filename}.cs"
        filepath.write_text(code, encoding='utf-8')
        
        # Parametreleri JSON olarak da kaydet
        params_path = self.output_dir / f"{filename}_params.json"
        params_path.write_text(json.dumps(params, indent=2, default=str), encoding='utf-8')
        
        return str(filepath)
    
    def export_all(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any],
        params3: Dict[str, Any],
        params4: Dict[str, Any],
        vade_tipi: str = "ENDEKS",
        lot_size: int = 1
    ) -> Dict[str, str]:
        """
        Tüm dosyaları tek seferde export eder.
        
        Returns:
            Dict with 'strategy1', 'strategy2', 'strategy3', 'strategy4', 'robot' paths
        """
        s1_path = self.export_strategy1(params1, vade_tipi)
        s2_path = self.export_strategy2(params2, vade_tipi)
        s3_path = self.export_strategy3(params3, vade_tipi)
        s4_path = self.export_strategy4(params4, vade_tipi)
        robot_path = self.export_combined_robot(lot_size)
        
        return {
            'strategy1': s1_path,
            'strategy2': s2_path,
            'strategy3': s3_path,
            'strategy4': s4_path,
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
