import React, { useState, useRef } from 'react';
import { Upload, Play, Download, FileVideo, Languages, Loader2, Wand2, Settings } from 'lucide-react';

export default function SeamlessVideoTranslator() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [sourceLanguage, setSourceLanguage] = useState('en');
  const [targetLanguage, setTargetLanguage] = useState('es');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processedVideo, setProcessedVideo] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced Controls State
  const [preserveStyle, setPreserveStyle] = useState(true);
  const [durationFactor, setDurationFactor] = useState(1.0);

  const fileInputRef = useRef(null);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'es', name: 'Spanish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'it', name: 'Italian' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'zh', name: 'Chinese' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'ar', name: 'Arabic' },
    { code: 'hi', name: 'Hindi' },
    { code: 'ru', name: 'Russian' },
  ];

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '24px',
      fontFamily: 'sans-serif',
    },
    mainCard: {
      maxWidth: '1024px',
      margin: '0 auto',
      backgroundColor: 'rgba(255, 255, 255, 0.98)',
      backdropFilter: 'blur(10px)',
      borderRadius: '24px',
      padding: '32px',
      boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
    },
    header: {
      textAlign: 'center',
      marginBottom: '32px',
    },
    headerTitle: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: '16px',
    },
    title: {
      fontSize: '2.5rem',
      fontWeight: 'bold',
      background: 'linear-gradient(135deg, #667eea, #764ba2)',
      backgroundClip: 'text',
      WebkitBackgroundClip: 'text',
      color: 'transparent',
      marginLeft: '12px',
    },
    subtitle: {
      fontSize: '1.25rem',
      color: '#666',
      marginBottom: '8px',
    },
    description: {
      color: '#888',
    },
    uploadArea: {
      border: '3px dashed #d1d5db',
      borderRadius: '16px',
      padding: '48px',
      textAlign: 'center',
      marginBottom: '32px',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      backgroundColor: '#f8f9ff',
    },
    uploadAreaHover: {
      borderColor: '#764ba2',
      backgroundColor: '#f0f2ff',
    },
    uploadContent: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
    },
    uploadText: {
      textAlign: 'left',
    },
    uploadTitle: {
      fontSize: '1.25rem',
      fontWeight: '600',
      color: '#333',
      marginBottom: '8px',
    },
    uploadSubtext: {
      color: '#666',
    },
    languageGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '24px',
      marginBottom: '24px',
    },
    languageCard: {
      backgroundColor: '#f9fafb',
      padding: '24px',
      borderRadius: '12px',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
    },
    label: {
      display: 'block',
      fontWeight: '600',
      color: '#333',
      marginBottom: '8px',
    },
    select: {
      width: '100%',
      padding: '12px',
      border: '2px solid #e5e7eb',
      borderRadius: '8px',
      fontSize: '1rem',
      transition: 'border-color 0.3s ease',
    },
    selectFocus: {
      borderColor: '#667eea',
      boxShadow: '0 0 0 3px rgba(102, 126, 234, 0.1)',
    },
    progressSection: {
      backgroundColor: 'white',
      padding: '20px',
      borderRadius: '10px',
      marginBottom: '20px',
      display: 'none',
    },
    progressSectionVisible: {
      display: 'block',
    },
    progressBar: {
      width: '100%',
      height: '10px',
      backgroundColor: '#e5e7eb',
      borderRadius: '5px',
      overflow: 'hidden',
      marginBottom: '10px',
    },
    progressFill: {
      height: '100%',
      background: 'linear-gradient(90deg, #667eea, #764ba2)',
      transition: 'width 0.3s ease',
    },
    progressText: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '16px',
    },
    progressLabel: {
      fontSize: '1.125rem',
      fontWeight: '600',
      color: '#374151',
    },
    progressPercent: {
      color: '#667eea',
      fontWeight: 'bold',
    },
    progressDescription: {
      textAlign: 'center',
      color: '#6b7280',
    },
    buttonContainer: {
      display: 'flex',
      gap: '16px',
      marginBottom: '32px',
    },
    translateButton: {
      flex: 1,
      background: 'linear-gradient(135deg, #667eea, #764ba2)',
      color: 'white',
      padding: '16px 32px',
      borderRadius: '12px',
      fontWeight: '600',
      fontSize: '1.125rem',
      border: 'none',
      cursor: 'pointer',
      transition: 'transform 0.3s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
    },
    translateButtonHover: {
      transform: 'translateY(-2px)',
      boxShadow: '0 10px 25px rgba(102, 126, 234, 0.3)',
    },
    translateButtonDisabled: {
      background: '#ccc',
      cursor: 'not-allowed',
      transform: 'none',
    },
    downloadButton: {
      backgroundColor: '#10b981',
      color: 'white',
      padding: '16px 32px',
      borderRadius: '12px',
      fontWeight: '600',
      fontSize: '1.125rem',
      border: 'none',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    downloadButtonHover: {
      backgroundColor: '#059669',
      transform: 'translateY(-2px)',
    },
    videoPreview: {
      backgroundColor: '#f9fafb',
      borderRadius: '12px',
      padding: '24px',
    },
    videoPreviewTitle: {
      fontSize: '1.25rem',
      fontWeight: '600',
      color: '#374151',
      marginBottom: '16px',
    },
    video: {
      width: '100%',
      maxWidth: '512px',
      margin: '0 auto',
      borderRadius: '8px',
      boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
    },
    hiddenInput: {
      display: 'none',
    },
    toggleSwitch: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '16px',
    },
    toggleLabel: {
      fontWeight: '500',
      color: '#555',
    },
    switch: {
      position: 'relative',
      display: 'inline-block',
      width: '60px',
      height: '34px',
    },
    switchInput: {
      opacity: 0,
      width: 0,
      height: 0,
    },
    slider: {
      position: 'absolute',
      cursor: 'pointer',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: '#ccc',
      transition: '.4s',
      borderRadius: '34px',
    },
    sliderBefore: {
      position: 'absolute',
      content: '""',
      height: '26px',
      width: '26px',
      left: '4px',
      bottom: '4px',
      backgroundColor: 'white',
      transition: '.4s',
      borderRadius: '50%',
    },
    advancedButton: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 16px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      background: 'white',
      cursor: 'pointer',
      marginBottom: '16px',
    },
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setProcessedVideo(null);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setProcessedVideo(null);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const translateVideo = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setProgress(0);

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('source_lang', sourceLanguage);
    formData.append('target_lang', targetLanguage);
    formData.append('preserve_style', preserveStyle);
    formData.append('duration_factor', durationFactor);

    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(90, prev + Math.random() * 10));
      }, 800);

      const response = await fetch('http://localhost:8004/translate', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);

      if (response.ok) {
        const blob = await response.blob();
        const videoUrl = URL.createObjectURL(blob);
        setProcessedVideo(videoUrl);
        setProgress(100);
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Translation failed. Check backend logs.' }));
        console.error('Translation failed:', response.status, errorData);
        alert(`Translation failed: ${errorData.detail || response.statusText}`);
        setProcessedVideo(null);
        setProgress(0);
      }
    } catch (error) {
      console.error('Error translating video:', error);
      alert('Translation failed. Please try again. Check console for details.');
      setProcessedVideo(null);
      setProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadVideo = () => {
    if (processedVideo) {
      const a = document.createElement('a');
      a.href = processedVideo;
      a.download = `translated_${selectedFile.name}`;
      document.body.appendChild(a);
a.click();
      document.body.removeChild(a);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.mainCard}>
        <div style={styles.header}>
          <div style={styles.headerTitle}>
            <h1 style={styles.title}>SeamlessExpressive</h1>
          </div>
          <p style={styles.subtitle}>AI-Powered Video Translation System For Master's Thesis</p>
        </div>

        <div
          style={styles.uploadArea}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            style={styles.hiddenInput}
          />
          
          {selectedFile ? (
            <div style={styles.uploadContent}>
              <FileVideo size={64} color="#667eea" />
              <div style={styles.uploadText}>
                <p style={styles.uploadTitle}>{selectedFile.name}</p>
                <p style={styles.uploadSubtext}>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
          ) : (
            <div>
              <Upload size={64} color="#a855f7" style={{ margin: '0 auto 16px' }} />
              <p style={styles.uploadTitle}>Drop your video here or click to browse</p>
              <p style={styles.uploadSubtext}>Supports MP4, AVI, MOV, and other video formats</p>
            </div>
          )}
        </div>

        <div style={styles.languageGrid}>
          <div style={styles.languageCard}>
            <label style={styles.label}>Source Language</label>
            <select
              value={sourceLanguage}
              onChange={(e) => setSourceLanguage(e.target.value)}
              style={styles.select}
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>

          <div style={styles.languageCard}>
            <label style={styles.label}>Target Language</label>
            <select
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
              style={styles.select}
            >
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button onClick={() => setShowAdvanced(!showAdvanced)} style={styles.advancedButton}>
          <Settings size={18} />
          {showAdvanced ? 'Hide' : 'Show'} Advanced Controls
        </button>

        {showAdvanced && (
          <div style={{...styles.languageCard, marginBottom: '24px', padding: '24px'}}>
            <h3 style={{marginTop: 0, marginBottom: '24px'}}>Expressive Controls</h3>
            <div style={styles.toggleSwitch}>
              <span style={styles.toggleLabel}>
                Preserve Vocal Style
                <p style={{fontSize: '0.8rem', color: '#888', margin: '4px 0 0'}}>ON for original voice, OFF for generic voice.</p>
              </span>
              <label style={styles.switch}>
                <input type="checkbox" checked={preserveStyle} onChange={() => setPreserveStyle(!preserveStyle)} style={styles.switchInput} />
                <span style={{...styles.slider, backgroundColor: preserveStyle ? '#667eea' : '#ccc' }}>
                  <span style={{...styles.sliderBefore, transform: preserveStyle ? 'translateX(26px)' : 'translateX(0px)'}} />
                </span>
              </label>
            </div>
            
            <div>
              <label style={styles.label}>Speech Rate: {durationFactor.toFixed(1)}x</label>
              <p style={{fontSize: '0.8rem', color: '#888', margin: '-4px 0 12px'}}>Control speed and pauses. Only affects Expressive model.</p>
              <input
                type="range"
                min="0.5"
                max="1.5"
                step="0.1"
                value={durationFactor}
                onChange={(e) => setDurationFactor(parseFloat(e.target.value))}
                disabled={!preserveStyle}
                style={{ width: '100%', cursor: preserveStyle ? 'pointer' : 'not-allowed' }}
              />
            </div>
          </div>
        )}

        {isProcessing && (
          <div style={{...styles.progressSection, ...styles.progressSectionVisible}}>
            <div style={styles.progressText}>
              <span style={styles.progressLabel}>Processing Video...</span>
              <span style={styles.progressPercent}>{Math.round(progress)}%</span>
            </div>
            <div style={styles.progressBar}>
              <div style={{...styles.progressFill, width: `${progress}%`}}></div>
            </div>
            <p style={styles.progressDescription}>
              Translating with {preserveStyle ? 'SeamlessExpressive' : 'SeamlessM4Tv2'} AI...
            </p>
          </div>
        )}

        <div style={styles.buttonContainer}>
          <button
            onClick={translateVideo}
            disabled={!selectedFile || isProcessing}
            style={{
              ...styles.translateButton,
              ...((!selectedFile || isProcessing) ? styles.translateButtonDisabled : {})
            }}
          >
            {isProcessing ? (
              <>
                <Loader2 size={24} style={{animation: 'spin 1s linear infinite'}} />
                <span>Translating...</span>
              </>
            ) : (
              <>
                <Play size={24} />
                <span>Translate Video</span>
              </>
            )}
          </button>

          {processedVideo && (
            <button
              onClick={downloadVideo}
              style={styles.downloadButton}
            >
              <Download size={24} />
              <span>Download</span>
            </button>
          )}
        </div>

        {processedVideo && (
          <div style={styles.videoPreview}>
            <h3 style={styles.videoPreviewTitle}>Translated Video</h3>
            <video
              src={processedVideo}
              controls
              style={styles.video}
            >
              Your browser does not support the video tag.
            </video>
          </div>
        )}
      </div>
    </div>
  );
}