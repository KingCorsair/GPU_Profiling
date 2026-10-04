import { createContext, useContext, useEffect, useId, useRef, useState } from 'react';
import type { Dispatch, ReactNode, SetStateAction } from 'react';
import { ArrowRight, ImagePlus, RotateCcw } from 'lucide-react';
import './token-demo.css';

export type DemoImage = { src: string; name: string; base64?: string };

// An original illustration, not an evaluation image or model-derived saliency map.
const sampleSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="576" height="576" viewBox="0 0 576 576">
<rect width="576" height="576" fill="#c7d6cf"/><rect x="42" y="38" width="492" height="337" rx="3" fill="#eef0e4"/>
<path d="M58 54h225v304H58zm241 0h219v304H299z" fill="#98b9b4"/>
<path d="M58 248q76-151 153-52t72 8v154H58zm241-53q102-61 219 40v123H299z" fill="#79958b"/>
<path d="M58 303q117-71 225-1v56H58zm241-23q125-87 219-30v108H299z" fill="#4f796e"/>
<circle cx="430" cy="118" r="37" fill="#f5e1a9"/><rect y="375" width="576" height="201" fill="#c29b77"/>
<path d="M0 416h576M0 483h576M0 548h576" stroke="#b18a68" stroke-width="2"/>
<ellipse cx="297" cy="471" rx="177" ry="31" fill="#a57e5e"/>
<path d="M126 395h332q-20 110-165 107-141 0-167-107" fill="#305567"/>
<ellipse cx="292" cy="397" rx="166" ry="37" fill="#477589"/>
<ellipse cx="292" cy="397" rx="145" ry="27" fill="#233f50"/>
<circle cx="220" cy="357" r="53" fill="#d88243"/><circle cx="320" cy="340" r="57" fill="#e7a253"/><circle cx="369" cy="389" r="48" fill="#c8703b"/>
<path d="M210 307q-20-46-58-33 15 45 62 43M324 284q34-42 65-23-19 40-65 36" fill="#3d6652"/>
<path d="m222 315-8-20m105-10 7-18m42 78 4-14" stroke="#614f37" stroke-width="5" stroke-linecap="round"/>
<path d="M191 349q4-21 22-27m81 9q3-22 23-33m34 85q4-15 15-20" fill="none" stroke="#f8c985" stroke-width="7" stroke-linecap="round" opacity=".6"/>
</svg>`;
const sampleImage: DemoImage = { src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(sampleSvg)}`, name: 'Oranges by the window · sample illustration' };

const ImageContext = createContext<[DemoImage, Dispatch<SetStateAction<DemoImage>>] | null>(null);
export function DemoImageProvider({ children }: { children: ReactNode }) {
  const value = useState<DemoImage>(sampleImage);
  return <ImageContext.Provider value={value}>{children}</ImageContext.Provider>;
}
export function useDemoImage() {
  const value = useContext(ImageContext);
  if (!value) throw new Error('Image preview requires DemoImageProvider.');
  return value;
}

// Stable center-weighted pattern for teaching only. It never inspects an image,
// ranks model tokens, or participates in inference or reported measurements.
const illustrativeOrder = Array.from({ length: 576 }, (_, i) => i).sort((a, b) => {
  const rank = (i: number) => ((i % 24 - 11.5) ** 2 + (Math.floor(i / 24) - 12.5) ** 2) + ((i * 137) % 97) * .6;
  return rank(a) - rank(b) || a - b;
});

export function TokenImage({ image, tokens, overlay = true, grid = true }: { image: DemoImage; tokens: number; overlay?: boolean; grid?: boolean }) {
  const retained = new Set(illustrativeOrder.slice(0, tokens));
  return <div className="token-image-frame">
    <svg className="token-image" viewBox="0 0 576 576" role="img" aria-label={`${image.name}. ${overlay ? `Illustrative overlay: ${tokens} of 576 cells retained; this is not the model’s selection.` : 'Original image, unchanged.'}`}>
      <image href={image.src} width="576" height="576" preserveAspectRatio="xMidYMid meet" />
      {Array.from({ length: 576 }, (_, i) => <rect key={i} x={i % 24 * 24} y={Math.floor(i / 24) * 24} width="24" height="24" fill={overlay && !retained.has(i) ? '#111c25' : 'transparent'} fillOpacity=".88" stroke={grid ? '#fff' : 'none'} strokeOpacity={overlay && !retained.has(i) ? '.07' : '.22'} strokeWidth=".8" />)}
    </svg>
  </div>;
}

export function ImagePicker({ image, onChange, disabled = false, onLoadingChange, maxBytes = 8 * 1024 * 1024 }: { image: DemoImage; onChange: (image: DemoImage) => void; disabled?: boolean; onLoadingChange?: (loading: boolean) => void; maxBytes?: number }) {
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const sequence = useRef(0);
  useEffect(() => () => { sequence.current += 1; }, []);
  async function loadImage(file?: File) {
    if (!file) return;
    const request = ++sequence.current;
    setError(''); setLoading(false); onLoadingChange?.(false);
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { setError('Choose a JPG, PNG, or WebP image.'); return; }
    if (file.size > maxBytes) { setError(`Choose an image smaller than ${Math.round(maxBytes / 1024 / 1024)} MB.`); return; }
    setLoading(true); onLoadingChange?.(true);
    try {
      const src = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(new Error('Could not read this image.')); reader.readAsDataURL(file); });
      const decoded = new Image(); decoded.src = src; await decoded.decode();
      if (decoded.naturalWidth * decoded.naturalHeight > 40_000_000) throw new Error('Choose an image under 40 megapixels.');
      if (request === sequence.current) onChange({ src, name: file.name, base64: src.slice(src.indexOf(',') + 1) });
    } catch (reason) { if (request === sequence.current) setError(reason instanceof Error && reason.message.startsWith('Choose') ? reason.message : 'This image could not be opened. Try another JPG, PNG, or WebP.'); }
    finally { if (request === sequence.current) { setLoading(false); onLoadingChange?.(false); } }
  }
  return <div className="image-picker">
    <div className="image-picker-actions"><button type="button" className="image-upload-button" disabled={disabled || loading} onClick={() => input.current?.click()}><ImagePlus size={16} />{loading ? 'Opening image…' : 'Upload your image'}</button>
      {image !== sampleImage && <button type="button" className="image-reset-button" disabled={disabled || loading} onClick={() => { sequence.current += 1; setError(''); onChange(sampleImage); }}><RotateCcw size={14} />Use sample</button>}
    </div>
    <input ref={input} className="sr-only" type="file" tabIndex={-1} accept="image/jpeg,image/png,image/webp" aria-label="Choose an image" disabled={disabled || loading} onChange={(event) => { void loadImage(event.target.files?.[0]); event.target.value = ''; }} />
    <p className="image-picker-note">JPG, PNG, WebP · up to {Math.round(maxBytes / 1024 / 1024)} MB. Preview stays in your browser.</p>
    {error && <p className="image-error" role="alert">{error}</p>}
  </div>;
}

export function TokenBudget({ tokens, onChange, disabled = false }: { tokens: number; onChange: (tokens: number) => void; disabled?: boolean }) {
  const id = useId();
  return <div className="token-budget">
    <div className="token-budget-label"><label htmlFor={id}>Image tokens retained</label><output htmlFor={id}><b>{tokens}</b> / 576</output></div>
    <input id={id} type="range" min="1" max="576" step="1" value={tokens} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} aria-valuetext={`${tokens} of 576 tokens retained, ${Math.round((576 - tokens) / 576 * 100)} percent removed${tokens === 576 ? ', unpruned baseline' : ''}`} />
    <div className="token-range-labels"><span>Less image detail</span><span>Full detail</span></div>
    <div className="token-presets" aria-label="Token budget presets">{[32, 64, 128, 288, 576].map((value) => <button type="button" key={value} disabled={disabled} aria-pressed={tokens === value} onClick={() => onChange(value)}>{value}{value === 576 && <small>Baseline</small>}</button>)}</div>
    <p className="token-budget-caption">{tokens === 576 ? 'Full detail · Unpruned baseline. No tokens removed.' : `Keep ${tokens} of 576 visual tokens · ${((576 - tokens) / 576 * 100).toFixed(1)}% removed.`}</p>
  </div>;
}

export default function TokenImageDemo({ showPlaygroundLink = true }: { showPlaygroundLink?: boolean }) {
  const [image, setImage] = useDemoImage();
  const [tokens, setTokens] = useState(128);
  const [grid, setGrid] = useState(true);
  return <section className="image-demo-section" aria-labelledby="image-demo-title">
    <div className="image-demo-heading"><div><p className="project-label">01 / See the idea</p><h2 id="image-demo-title" tabIndex={-1}>Same image. Less information.</h2><p>Move the slider to see what keeping fewer image tokens could look like.</p></div></div>
    <div className="image-demo-workspace">
      <div className="image-demo-visuals">
        <figure><figcaption><span>Original image</span><b>576 tokens</b></figcaption><TokenImage image={image} tokens={576} overlay={false} grid={grid} /><p>Full detail · Unpruned baseline</p></figure>
        <figure><figcaption><span>Illustrative selection</span><b>{tokens} tokens</b></figcaption><TokenImage image={image} tokens={tokens} grid={grid} /><p>{576 - tokens} cells dimmed · {tokens} kept visible</p></figure>
        <div className="image-demo-toolbar"><ImagePicker image={image} onChange={setImage} /><label className="grid-toggle"><input type="checkbox" checked={grid} onChange={(event) => setGrid(event.target.checked)} />Show grid</label></div>
      </div>
      <div className="image-demo-controls"><TokenBudget tokens={tokens} onChange={setTokens} /><div className="image-demo-explanation"><h3>What changes inside the model?</h3><p>The image becomes a grid of 24 × 24 visual tokens. Pruning reduces the information passed to the language model.</p><p>The picture itself stays unchanged. The dimmed squares explain the idea; they are not a reconstructed image or the model’s actual selection.</p></div>{showPlaygroundLink && <a className="text-action" href="#/playground">Try the Playground <ArrowRight size={16} /></a>}</div>
    </div>
    <p className="image-demo-footnote">Illustration only: a fixed, center-weighted pattern is used for every image. Actual retained tokens and merged context require model trace data; this preview does not measure accuracy or speed.</p>
  </section>;
}
