const API = "https://zarkos.hidenplay.net/v1/chat/completions";
const KEY = "zarkos-0sIrN2b9a-api";

let temperature = 0.8;

async function chat(msg, temp) {
  const res = await fetch(API, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": KEY },
    body: JSON.stringify({
      model: "zarkos-mini-v1",
      messages: [{ role: "user", content: msg }],
      max_tokens: 100,
      temperature: temp,
    }),
  });
  const data = await res.json();
  if (data.error) return `[ERROR] ${data.error.message || JSON.stringify(data.error)}`;
  return data.choices[0].message.content;
}

const readline = require("readline").createInterface({
  input: process.stdin, output: process.stdout, prompt: "You: ",
});

console.log(`Connected. temperature=${temperature}`);
console.log("Commands:");
console.log("  /temp 0.2          -> change temperature (low = more consistent/deterministic)");
console.log("  /repeat 5 2+2?     -> send the same message 5 times, show all replies");
console.log("  exit               -> quit");
readline.prompt();

readline.on("line", async (line) => {
  const msg = line.trim();

  if (msg === "exit") return readline.close();

  if (msg.startsWith("/temp ")) {
    const val = parseFloat(msg.split(" ")[1]);
    if (isNaN(val)) {
      console.log("Usage: /temp 0.2");
    } else {
      temperature = val;
      console.log(`temperature set to ${temperature}`);
    }
    return readline.prompt();
  }

  if (msg.startsWith("/repeat ")) {
    const parts = msg.split(" ");
    const count = parseInt(parts[1]);
    const prompt = parts.slice(2).join(" ");
    if (isNaN(count) || !prompt) {
      console.log("Usage: /repeat 5 2+2?");
      return readline.prompt();
    }
    console.log(`Sending "${prompt}" ${count} times at temperature=${temperature}...`);
    for (let i = 1; i <= count; i++) {
      const reply = await chat(prompt, temperature);
      console.log(`  [${i}] ${reply}`);
    }
    return readline.prompt();
  }

  const reply = await chat(msg, temperature);
  console.log("Bot:", reply);
  readline.prompt();
});
