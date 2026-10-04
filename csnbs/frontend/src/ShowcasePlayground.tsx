import TokenImageDemo from './TokenImageDemo';

/** The static sharing build has no model client, gateway, or startup effects. */
export default function ShowcasePlayground() {
  return <div className="playground-page">
    <section className="intro" aria-labelledby="showcase-title">
      <p className="study-label">Interactive explanation</p>
      <h1 id="showcase-title">Explore image tokens.</h1>
      <p className="intro-copy">Explore the idea in your browser, then inspect the experiments and the records behind them.</p>
      <p className="figure-note">This version does not run a model. The image overlay is illustrative; actual measurements are in the study pages.</p>
      <div className="hero-actions"><a className="primary-action" href="#/october">View measured results</a><a className="text-action" href="#/archive">Explore run records →</a></div>
    </section>
    <TokenImageDemo showPlaygroundLink={false} />
  </div>;
}
