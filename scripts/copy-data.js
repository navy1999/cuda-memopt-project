#!/usr/bin/env node
/**
 * Copy ../results and ../report into public/ so they are available at build/deploy.
 * Run before next build when deploying (e.g. from repo root with root dir = frontend).
 */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const publicDir = path.join(root, "public");
const resultsSrc = path.join(root, "cuda", "results");
const reportSrc = path.join(root, "cuda", "report");
const resultsDest = path.join(publicDir, "results");
const reportDest = path.join(publicDir, "report");

function copyDir(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn("copy-data: source not found:", src);
    return;
  }
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src)) {
    const s = path.join(src, name);
    const d = path.join(dest, name);
    if (fs.statSync(s).isDirectory()) {
      copyDir(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}

fs.mkdirSync(publicDir, { recursive: true });
copyDir(resultsSrc, resultsDest);
copyDir(reportSrc, reportDest);
console.log("copy-data: copied results/ and report/ into public/");
