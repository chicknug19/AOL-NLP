import React, { useState } from 'react';

export default function HoaxDetector() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {
      setError('Teks berita tidak boleh kosong.');
      return;
    }

    setLoading(true);
    setResult(null);
    setError('');

    try {
      const response = await fetch('https://chicknug19-aol-nlp.hf.space/api/deteksi', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Kita kunci (hardcode) pengiriman datanya menggunakan model indobert
        body: JSON.stringify({ teks: text, model: 'indobert' }),
      });

      if (!response.ok) {
        throw new Error('Terjadi kesalahan pada server.');
      }

      const data = await response.json();
      
      setResult({
        status: data.status,
        label: data.analisis_ai.prediksi,
        confidence: data.analisis_ai.keyakinan / 100, 
        keterangan: data.analisis_ai.keterangan,
        referensi: data.cek_fakta_internet 
      });
    } catch (err) {
      console.error(err);
      setError('Gagal terhubung ke server AI. Pastikan backend Python sudah menyala.');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 relative overflow-hidden font-sans text-slate-800 selection:bg-teal-200 selection:text-teal-900 z-0">
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] rounded-full bg-gradient-to-br from-teal-300/40 to-blue-400/40 blur-[100px]"></div>
        <div className="absolute top-[20%] -right-[10%] w-[60%] h-[60%] rounded-full bg-gradient-to-bl from-violet-300/40 to-purple-400/40 blur-[120px]"></div>
        <div className="absolute -bottom-[20%] left-[20%] w-[50%] h-[50%] rounded-full bg-gradient-to-tr from-indigo-300/30 to-blue-300/30 blur-[100px]"></div>
      </div>
      
      <header className="bg-white/50 backdrop-blur-xl shadow-sm border-b border-white/60 sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-6 py-3 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="p-1">
                <img 
                    src="src/assets/logo.png" 
                    alt="Smart Hoax Detector Logo" 
                    className="h-10 w-10 object-contain"
                />
            </div>
            <div className='flex flex-col'>
                <h1 className="text-xl md:text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-700 via-blue-700 to-indigo-800 tracking-tight leading-none">
                  Smart Hoax Detector
                </h1>
                <span className='text-xs text-slate-600 font-bold'>Intelligent Fake News Detection</span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-700 bg-white/60 backdrop-blur-md px-4 py-2 rounded-full border border-white/80 shadow-sm">
             <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse"></span>
             Kelompok 1 • Project NLP
          </div>
        </div>
      </header>
      
      <main className="max-w-5xl mx-auto px-4 py-10 md:py-16">
        <div className="w-full rounded-3xl mb-10 overflow-hidden shadow-2xl shadow-blue-900/10 border-4 border-white/80">
            <img 
                src="src/assets/bannerhoax.jpg" 
                alt="Banner Ilustrasi Deteksi Hoaks" 
                className="w-full h-48 md:h-72 object-cover" 
            />
        </div>
        
        <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-2xl shadow-indigo-900/10 p-8 md:p-12 border border-white/70 relative overflow-hidden">      
          <div className="relative z-10">
            <div className="text-center mb-10">
                <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">Verifikasi Fakta Berita</h2>
                <p className="text-slate-700 text-lg leading-relaxed max-w-3xl mx-auto font-medium">
                Silahkan memeriksa keaslian dari informasi yang telah ditemukan. Tempelkan teks artikel atau berita yang mencurigakan di bawah, dan biarkan AI IndoBERT kami menganalisis pola bahasanya.
                </p>
            </div>
            <hr className="border-slate-300/50 mb-10" />
            
            <form onSubmit={handleSubmit} className="space-y-8">
              <div className="w-full">
                  <label htmlFor="news-text" className="block text-base font-bold text-slate-800 mb-3 hover:text-blue-800 transition-colors">
                    Konten Berita untuk Dianalisis
                  </label>
                  <div className="relative group">
                    <textarea
                      id="news-text"
                      rows="8"
                      value={text}
                      onChange={(e) => {
                        setText(e.target.value);
                        if (error) setError('');
                      }}
                      placeholder="Contoh: 'Telah terjadi gempa bumi berkekuatan...'"
                      className="w-full p-6 bg-white/80 backdrop-blur-sm border-2 border-slate-200/80 rounded-2xl focus:ring-2 focus:ring-blue-300 focus:border-blue-500 focus:bg-white outline-none transition-all resize-y text-slate-800 placeholder-slate-400 shadow-inner font-medium leading-relaxed"
                    ></textarea>
                    
                    <div className="absolute bottom-4 right-5 text-xs font-bold text-slate-500 bg-white/90 px-2 py-1 rounded shadow-sm border border-slate-100">
                      {text.length.toLocaleString()} karakter
                    </div>
                  </div>
                  {error && (
                    <p className="mt-4 text-sm font-bold text-red-600 flex items-center bg-red-100/80 backdrop-blur-sm p-3 rounded-lg border border-red-200 animate-pulse">
                      <svg className="w-5 h-5 mr-2.5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd"></path></svg>
                      {error}
                    </p>
                  )}
              </div>
              
              <div className="flex justify-center pt-2">
                <button
                  type="submit"
                  disabled={loading}
                  className={`w-full md:w-2/3 lg:w-1/2 flex justify-center items-center gap-2.5 py-4 px-10 rounded-xl text-white font-extrabold text-lg transition-all duration-300
                    ${loading 
                      ? 'bg-slate-400 cursor-wait' 
                      : 'bg-gradient-to-r from-teal-500 via-blue-600 to-violet-600 hover:from-teal-600 hover:via-blue-700 hover:to-violet-700 shadow-lg shadow-blue-500/20 hover:shadow-violet-500/40 hover:-translate-y-1 active:translate-y-0 active:shadow-md'}`}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Memproses AI...
                    </>
                  ) : (
                    <>
                      Mulai Deteksi dengan IndoBERT
                      <svg className="w-5 h-5 ml-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Result Area */}
            {result && !loading && (
              <div className="mt-12 pt-10 border-t-4 border-dashed border-slate-300/50 animate-fade-in-up">
                <div className={`relative overflow-hidden p-10 rounded-3xl border-2 transition-all shadow-2xl backdrop-blur-md mb-8 ${
                  result.label === 'HOAKS' 
                    ? 'bg-red-50/80 border-red-200 shadow-red-900/10' 
                    : 'bg-emerald-50/80 border-emerald-200 shadow-emerald-900/10'
                }`}>
                  
                  <div className="relative z-10 flex flex-col items-center text-center">
                    <span className="text-sm font-extrabold uppercase tracking-widest text-slate-600 mb-3 bg-white/80 backdrop-blur-sm px-4 py-1.5 rounded-full shadow-inner border border-white">Kesimpulan Analisis</span>                  
                    <h3 className={`text-6xl md:text-7xl font-black mb-4 tracking-tighter ${
                      result.label === 'HOAKS' ? 'text-red-600' : 'text-emerald-600'
                    }`}>
                      {result.label}
                    </h3>
                    <p className="text-slate-600 font-medium max-w-2xl mb-8">
                      {result.keterangan}
                    </p>
                    
                    <div className="w-full max-w-xl bg-white/90 p-6 rounded-2xl border border-white shadow-lg backdrop-blur-sm">
                      <div className="flex justify-between text-base mb-2.5 font-bold text-slate-800">
                        <span>Tingkat Keyakinan Model</span>
                        <span className={`px-3 py-1 rounded-md ${result.label === 'HOAKS' ? 'bg-red-100 text-red-800' : 'bg-emerald-100 text-emerald-800'}`}>
                          {(result.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-5 overflow-hidden shadow-inner border border-slate-300/50 relative">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ease-out relative z-10 ${
                            result.label === 'HOAKS' 
                              ? 'bg-gradient-to-r from-red-400 to-red-600' 
                              : 'bg-gradient-to-r from-emerald-400 to-emerald-600'
                          }`}
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-slate-600 mt-4 text-center font-bold bg-slate-100/50 py-2 rounded-lg border border-slate-200">
                        Dianalisis menggunakan model <strong className="text-teal-700 uppercase">INDOBERT</strong> pada jaringan NLP Kelompok 1.
                      </p>
                    </div>
                  </div>
                </div>

                {/* UI Cek Fakta (RAG Hybrid) */}
                {result.referensi && result.referensi.length > 0 && (
                  <div className="mt-8">
                    <h4 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                      <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                      Cek Fakta Internet (Referensi Terkait)
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {result.referensi.map((ref, index) => (
                        <a key={index} href={ref.link} target="_blank" rel="noopener noreferrer" className="bg-white p-5 rounded-2xl shadow-md border border-slate-200 hover:shadow-xl hover:border-blue-300 transition-all group flex flex-col h-full">
                          <h5 className="font-bold text-slate-800 group-hover:text-blue-700 line-clamp-2 mb-3">{ref.judul}</h5>
                          <p className="text-sm text-slate-600 line-clamp-3 mb-4 flex-grow">{ref.cuplikan}</p>
                          <span className="text-xs font-semibold text-blue-600 mt-auto flex items-center gap-1">
                            Baca artikel asli
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
                          </span>
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
      
      <footer className="relative z-10 max-w-6xl mx-auto px-6 py-8 mt-4 text-center text-sm font-semibold text-slate-500 border-t border-slate-300/40">
        <p>&copy; 2026 Project Smart Hoax Detector. All rights reserved.</p>
      </footer>
    </div>
  );
}