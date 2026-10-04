// Demo server: all interfaces, port 8080 (override with PORT=). "/" serves
// the generated hub (build/hub.html); every generated artifact lives under
// ./build/ and is looked up there first, so /out.js etc. keep short URLs. /js/* serves the npm package sources from ../js/*
// — the hub's pyodide engine imports the tracer half (js/src/trace/) from
// there, so what runs is exactly what the package ships. Collects /report
// so a page opened with ?report=1 posts its results straight back to this
// box (no copy-paste) — the e2e suite (tests/test_webgpu_export.py ) reads them from /results.
const results = {};
const port = Number(Bun.env.PORT ?? 8080);
Bun.serve({
  port,
  hostname: "0.0.0.0",
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === 'POST' && url.pathname === '/report') {
      const body = await req.json();
      results[Date.now()] = body;
      console.log('REPORT received');
      return new Response('ok', { headers: { 'Access-Control-Allow-Origin': '*' } });
    }
    if (url.pathname === '/results') return Response.json(results);
    let path = url.pathname.slice(1) || 'build/hub.html';
    if (path.split('/').includes('..')) return new Response('nope', { status: 403 });
    if (path.startsWith('js/')) path = '../../js/' + path.slice(3);
    else if (!path.startsWith('build/')) {
      // generated artifacts win over same-named sources (there are none, but
      // the rule keeps the URLs stable if that ever changes)
      const built = Bun.file('build/' + path);
      if (await built.exists()) return new Response(built, { headers: { 'Cache-Control': 'no-store' } });
    }
    const file = Bun.file(path);
    if (!(await file.exists())) return new Response('nope', { status: 404 });
    return new Response(file, { headers: { 'Cache-Control': 'no-store' } });
  },
});
console.log(`demo server on 0.0.0.0:${port} — "/" is the hub (?report=1 auto-reports here)`);
