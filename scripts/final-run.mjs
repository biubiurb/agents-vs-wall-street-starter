import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import XLSX from "xlsx";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const challenge = JSON.parse(
  await fs.readFile(path.join(root, "challenge", "companies.json"), "utf8"),
);
const asOf = process.env.FORECAST_AS_OF || "2026-08-16";
const python = process.env.PYTHON || "python3";
const runStamp = new Date().toISOString().replaceAll(":", "-");
const runDirectory = path.join(root, "data", "final-run");
const submissionDirectory = path.join(root, "submission");
const logDirectory = path.join(root, "logs");
const logPath = path.join(logDirectory, `final-run-${runStamp}.log`);

await Promise.all([
  fs.mkdir(runDirectory, { recursive: true }),
  fs.mkdir(submissionDirectory, { recursive: true }),
  fs.mkdir(logDirectory, { recursive: true }),
]);

async function log(message = "") {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  process.stdout.write(line);
  await fs.appendFile(logPath, line, "utf8");
}

function summaryHeaderRow(sheet, period) {
  for (let row = 1; row <= 30; row += 1) {
    const value = (column) =>
      sheet[XLSX.utils.encode_cell({ r: row - 1, c: column - 1 })]?.v;
    if (
      String(value(1) ?? "").trim() === "Metric" &&
      String(value(2) ?? "").trim() === "Units" &&
      String(value(3) ?? "").trim() === period
    ) {
      return row;
    }
  }
  throw new Error(`Could not find the Metric / Units / ${period} header`);
}

await log(`SignalBridge clear run started; cutoff=${asOf}; companies=${challenge.companies.length}`);

for (const company of challenge.companies) {
  const slug = path.basename(company.outputFile, ".xlsx");
  const forecastPath = path.join(runDirectory, `${slug}.json`);
  const auditPath = path.join(runDirectory, `${slug}-audit.json`);
  const args = [
    "-m",
    "forecast_agents.main",
    "--company",
    company.ticker,
    "--period",
    company.period,
    "--as-of",
    asOf,
    "--details-output",
    auditPath,
    "--output",
    forecastPath,
  ];

  await log(`START ${company.ticker} ${company.period}`);
  const result = spawnSync(python, args, {
    cwd: root,
    encoding: "utf8",
    env: process.env,
    maxBuffer: 16 * 1024 * 1024,
  });
  if (result.stdout?.trim()) await log(`${company.ticker} stdout: ${result.stdout.trim()}`);
  if (result.stderr?.trim()) await log(`${company.ticker} stderr: ${result.stderr.trim()}`);
  if (result.error || result.status !== 0) {
    await log(`FAIL ${company.ticker}: ${result.error?.message || `exit ${result.status}`}`);
    throw result.error || new Error(`${company.ticker} forecast failed with exit ${result.status}`);
  }

  const forecasts = JSON.parse(await fs.readFile(forecastPath, "utf8"));
  const templatePath = path.join(root, "challenge", "templates", company.outputFile);
  const outputPath = path.join(submissionDirectory, company.outputFile);
  const workbook = XLSX.readFile(templatePath, { cellStyles: true });
  const sheet = workbook.Sheets.Summary;
  if (!sheet) throw new Error(`${company.outputFile} template has no Summary sheet`);
  const headerRow = summaryHeaderRow(sheet, company.period);

  for (const [index, metric] of company.metrics.entries()) {
    const forecast = forecasts[metric.label];
    if (!forecast || typeof forecast.predicted_value !== "number" || !Number.isFinite(forecast.predicted_value)) {
      throw new Error(`${company.ticker} has no finite forecast for ${metric.label}`);
    }
    if (forecast.unit !== metric.units) {
      throw new Error(
        `${company.ticker} returned ${forecast.unit || "no unit"} for ${metric.label}; expected ${metric.units}`,
      );
    }
    const address = XLSX.utils.encode_cell({ r: headerRow + index, c: 2 });
    sheet[address] = { ...(sheet[address] || {}), t: "n", v: forecast.predicted_value };
    await log(`VALUE ${company.ticker} | ${metric.label} | ${forecast.predicted_value} ${metric.units}`);
  }

  XLSX.writeFile(workbook, outputPath, { compression: true, cellStyles: true });
  await log(`DONE ${company.ticker}: ${path.relative(root, outputPath)}`);
}

const check = spawnSync(process.execPath, [path.join(root, "scripts", "check-forecasts.mjs")], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 4 * 1024 * 1024,
});
if (check.stdout?.trim()) await log(`CHECK stdout: ${check.stdout.trim()}`);
if (check.stderr?.trim()) await log(`CHECK stderr: ${check.stderr.trim()}`);
if (check.error || check.status !== 0) {
  await log(`FAIL workbook checks: ${check.error?.message || `exit ${check.status}`}`);
  throw check.error || new Error(`Workbook checks failed with exit ${check.status}`);
}

await log(`SUCCESS all four workbooks passed shape checks; log=${path.relative(root, logPath)}`);
