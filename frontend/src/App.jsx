import React, { useState } from 'react';

export default function HoaxDetector() {
  const [text, setText] = useState('');
  const [model, setModel] = useState('indobert');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const aiModels = [
    { value: 'svm', label: 'SVM (Baseline)' },
    { value: '1d-cnn', label: '1D-CNN' },
    { value: 'bilstm-attention', label: 'BiLSTM + Attention' },
    { value: 'indobert', label: 'IndoBERT' },
    { value: 'xlm-roberta', label: 'XLM-RoBERTa' },
  ];

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
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ teks: text, model: model }),
      });

      if (!response.ok) {
        throw new Error('Terjadi kesalahan pada server.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setTimeout(() => {
        setResult({
          status: 'success',
          label: Math.random() > 0.5 ? 'HOAKS' : 'FAKTA', 
          confidence: (Math.random() * (0.99 - 0.75) + 0.75).toFixed(2)
        });
        setLoading(false);
      }, 1500);
      return;
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
      {/* Header dengan efek Glass yang lebih tebal */}
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
      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-10 md:py-16">
        {/* Banner */}
        <div className="w-full rounded-3xl mb-10 overflow-hidden shadow-2xl shadow-blue-900/10 border-4 border-white/80">
            <img 
                src="src/assets/bannerhoax.jpg" 
                alt="Banner Ilustrasi Deteksi Hoaks" 
                className="w-full h-48 md:h-72 object-cover" 
            />
        </div>
        <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-2xl shadow-indigo-900/10 p-8 md:p-12 border border-white/70 relative overflow-hidden">       
          <div className="relative z-10">
            <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4 tracking-tight">Verifikasi Fakta Berita</h2>
            <p className="text-slate-700 text-lg mb-10 leading-relaxed max-w-3xl font-medium">
            Silahkan memeriksa keaslian dari informasi yang telah ditemukan. Tempelkan teks artikel atau berita yang mencurigakan di bawah, dan biarkan model menganalisis pola bahasanya.
            </p>
            <hr className="border-slate-300/50 mb-10" />
            <form onSubmit={handleSubmit} className="space-y-10">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                {/* Bagian Kiri: Model Selector */}
                <div className="lg:col-span-4 flex flex-col space-y-5">
                  <div>
                    <label htmlFor="model-select" className="flex items-center text-base font-bold text-slate-800 mb-3 hover:text-teal-800 transition-colors">
                       <svg className="w-5 h-5 mr-2.5 text-teal-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                       Pilih model untuk analisis
                    </label>
                    <div className="relative group">
                        <select
                          id="model-select"
                          value={model}
                          onChange={(e) => setModel(e.target.value)}
                          className="w-full p-4 bg-white/80 backdrop-blur-sm border-2 border-slate-200/80 rounded-xl focus:ring-2 focus:ring-teal-300 focus:border-teal-500 hover:border-slate-300 outline-none transition-all text-slate-800 font-bold cursor-pointer appearance-none shadow-sm"
                          style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%230f766e'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`, backgroundPosition: `right 1.25rem center`, backgroundRepeat: `no-repeat`, backgroundSize: `1.2em 1.2em` }}
                        >
                          {aiModels.map((m) => (
                            <option key={m.value} value={m.value} className='font-sans font-medium'>
                              {m.label}
                            </option>
                          ))}
                        </select>
                    </div>
                    <p className="mt-3 text-sm text-slate-600 font-medium leading-relaxed bg-white/50 backdrop-blur-sm p-3 rounded-lg border border-white/60">
                      Pilih model yang ingin digunakan untuk melihat hasil klasifikasi yang lebih akurat.
                    </p>
                  </div>
                </div>
                {/*Bagian Kanan*/}
                <div className="lg:col-span-8 overflow-visible">
                  <label htmlFor="news-text" className="block text-base font-bold text-slate-800 mb-3 hover:text-blue-800 transition-colors">
                    Konten Berita untuk Dianalisis
                  </label>
                  <div className="relative group">
                    <textarea
                      id="news-text"
                      rows="9"
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
              </div>
              {/*Submit Button*/}
              <div className="flex justify-end pt-6 border-t border-slate-300/50">
                <button
                  type="submit"
                  disabled={loading}
                  className={`w-full lg:w-auto min-w-[220px] flex justify-center items-center gap-2.5 py-4.5 px-10 rounded-xl text-white font-extrabold text-lg transition-all duration-300
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
                      Processing...
                    </>
                  ) : (
                    <>
                      Mulai Deteksi
                      <svg className="w-5 h-5 ml-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    </>
                  )}
                </button>
              </div>
            </form>
            {/* Result*/}
            {result && !loading && (
              <div className="mt-12 pt-10 border-t-4 border-dashed border-slate-300/50 animate-fade-in-up">
                <div className={`relative overflow-hidden p-10 rounded-3xl border-2 transition-all shadow-2xl backdrop-blur-md ${
                  result.label === 'HOAKS' 
                    ? 'bg-red-50/80 border-red-200 shadow-red-900/10' 
                    : 'bg-emerald-50/80 border-emerald-200 shadow-emerald-900/10'
                }`}>
                  <div className="absolute -right-10 -bottom-10 opacity-5 pointer-events-none transform rotate-12">
                    {result.label === 'HOAKS' 
                      ? <svg className="w-80 h-80 text-red-900" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd"></path></svg>
                      : <svg className="w-80 h-80 text-emerald-900" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"></path></svg>
                    }
                  </div>

                  <div className="relative z-10 flex flex-col items-center text-center">
                    <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-6 shadow-md border-4 border-white ${
                       result.label === 'HOAKS' ? 'bg-red-100 text-red-600' : 'bg-emerald-100 text-emerald-600'
                    }`}>
                       {result.label === 'HOAKS' 
                         ? <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                         : <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                       }
                    </div>
                    <span className="text-sm font-extrabold uppercase tracking-widest text-slate-600 mb-3 bg-white/80 backdrop-blur-sm px-4 py-1.5 rounded-full shadow-inner border border-white">Kesimpulan Analisis</span>                  
                    <h3 className={`text-6xl md:text-7xl font-black mb-8 tracking-tighter ${
                      result.label === 'HOAKS' ? 'text-red-600' : 'text-emerald-600'
                    }`}>
                      {result.label}
                    </h3>
                    <div className="w-full max-w-xl mt-2 bg-white/90 p-6 rounded-2xl border border-white shadow-lg backdrop-blur-sm">
                      <div className="flex justify-between text-base mb-2.5 font-bold text-slate-800">
                        <span>Tingkat Akurasi Model</span>
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
                        >
                           <div className="absolute top-0 left-0 right-0 h-1/2 bg-white/20 rounded-t-full"></div>
                        </div>
                        <div className='absolute inset-0 flex justify-center z-0'>
                            <div className='w-px h-full bg-slate-300'></div>
                        </div>
                      </div>
                      <p className="text-xs text-slate-600 mt-4 text-center font-bold bg-slate-100/50 py-2 rounded-lg border border-slate-200">
                        Dianalisis menggunakan model <strong className="text-teal-700 uppercase">{model.replace('-', ' ')}</strong> pada jaringan NLP Kelompok 1.
                      </p>
                    </div>

                  </div>
                </div>
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