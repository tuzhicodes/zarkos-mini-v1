# Tiny LLM (from scratch, pure Node.js)

Ek real decoder-only transformer (GPT-style), pure `@tensorflow/tfjs-node` tensor ops se
likha gaya (Layers API nahi, sab weights khud ke `tf.Variable` hain) — isliye ye
"real" chhota LLM hai, kisi pre-built model ka wrapper nahi.

## Setup

```bash
npm install
```

> Sandbox mein network nahi tha isliye `npm install` yahin nahi chalaya gaya —
> apne local machine/hosting pe ye zaroor chalayein.

## 1. Apna data daalo

`data/` folder mein jitni chaho utni `.jsonl` files daal do. Har line ek conversation:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}]}
```

Ek example file (`data/example.jsonl`) already di hui hai — real training se pehle
usse zyada bada, apna real dataset use karo (jitna zyada data, utna behtar chatbot).

## 2. Train karo

```bash
npm run train
```

Pehli baar chalane pe ye:
1. Sab `.jsonl` files se ek BPE tokenizer train karega aur `checkpoint/tokenizer.json` mein save karega.
2. Model ko `data/`se banaye gaye windows pe train karega.
3. Har epoch ke baad `checkpoint/weights.bin` + `checkpoint/manifest.json` save karega.

Dubara `npm run train` chalane pe existing checkpoint se resume hoga.

## 3. Chat karo

**Terminal mein:**
```bash
npm run chat
```

**Web UI mein:**
```bash
npm run serve
```
Fir browser mein `http://localhost:3000` kholo. UI mein do tabs hain:
- **Chat** — model se live baat karke test karo.
- **Train** — yahin se training **Start**/**Stop** kar sakte ho, live logs dekh sakte ho.
  Training ek alag background process mein chalti hai (server freeze nahi hota), aur
  har epoch ke baad checkpoint save hota hai — isliye beech mein **Stop** karna safe hai.
  Training khatam/stop hone ke baad Chat tab apne aap latest checkpoint reload kar leta hai.

## Config badalna ho to

`.env` file mein `N_LAYER`, `N_EMBD`, `N_HEAD`, `CONTEXT_LENGTH`, `VOCAB_SIZE` waghera
change kar sakte ho. Defaults ~5-10M parameter range mein hain jo CPU pe reasonable
time mein train ho jaate hain.

⚠️ **Config badalne ke baad purana checkpoint kaam nahi karega** (shapes mismatch ho
jaayengi) — `checkpoint/` folder khali karke dubara train karo.

## Folder structure

```
index.js          - CLI: train / serve / chat
model/
  config.js        - hyperparameters (.env se load)
  tokenizer.js      - from-scratch BPE tokenizer
  dataset.js        - multi-jsonl loader + chat template + windowing
  transformer.js     - TinyGPT: decoder-only transformer (raw tfjs ops)
  train.js          - training loop
  generate.js        - sampling / inference
  checkpoint.js      - binary weight save/load
checkpoint/         - tokenizer.json, manifest.json, weights.bin (training se banenge)
data/               - aapki *.jsonl training files
ui/
  server.js         - Express API + static server + training start/stop endpoints
  main.html         - mobile-app style dark UI: Chat tab + Train tab (start/stop + live logs)
```

## Fixed bugs (is session mein)

- **`dataDir`/`checkpointDir` CWD-dependent the**: `model/config.js` mein ye
  `'./data'` / `'./checkpoint'` (relative to jahan se process launch hui) the — jabki
  `.env` loading aur `HOST` binding already project-root-anchored fix ho chuke the.
  Ab `path.resolve(projectRoot, ...)` use hota hai, isliye `node index.js train` ya
  `npm start` kahin se bhi chalao (dusre folder se, cron se, hosting panel se),
  hamesha sahi `data/`/`checkpoint/` folder use hoga. Agar `.env` mein absolute path
  do (`DATA_DIR=/some/abs/path`), wo bhi respect hota hai.
- **UI ↔ index.js connect nahi tha properly ek jagah se sab kaam ho, waisa** — ab
  `npm start`/`node index.js` (bina args) `.env`'s `START_MODE` follow karke seedha
  web server boot karta hai, aur UI se hi training bhi control ho jaati hai.

## Fixed bugs (is round mein)

- **Training beech mein SIGKILL ho ke ruk jaati thi (out-of-memory)**: bade dataset
  (jaise 70k+ conversations) ke saath default `BATCH_SIZE=16`/`CONTEXT_LENGTH=128`
  memory-limited hosting (jaise Discord-bot panels, jo aksar 256MB-1GB hi dete hai)
  par host ke OOM killer se crash ho jaate the. Do fixes:
  1. Defaults ab safer hai (`BATCH_SIZE=4`, `CONTEXT_LENGTH=64`).
  2. **Automatic recovery**: agar training phir bhi `SIGKILL` se maregi (checkpoint
     save hone se pehle), server khud batch size aadha kar ke (aur zaroorat pade to
     context length bhi) training **automatically restart** kar dega — 4 attempts
     tak. Train tab ke logs mein ye clearly dikhega ("Auto-retrying with a smaller
     BATCH_SIZE..."). Ek baar checkpoint save ho jaaye to ye adjust karna band ho
     jaata hai (taaki weights ki shape checkpoint se mismatch na ho).
  - Agar 4 attempts ke baad bhi OOM hoti rahe, `.env` mein `N_EMBD`/`N_LAYER`/
    `FFN_HIDDEN` chhote karo (chhota model) ya apna dataset chhote batches mein
    train karo, ya zyada RAM wale host pe move karo.

## Fixed bugs (pichle round mein)

- **`server.js` khud publicly serve ho raha tha**: `express.static(__dirname)` poora
  `ui/` folder serve karta tha, jisme `server.js` bhi tha — koi bhi
  `http://yourdomain/server.js` khol ke tumhara server source dekh sakta tha.
  Ab `main.html` `ui/public/` mein hai aur sirf wahi folder static serve hota hai.
- **Crash hone par kuch dikhta hi nahi tha**: agar koi unexpected error aata (bad
  request, fs error, waghera), poora Node process silently mar jaata — hosting panel
  pe bilkul "website nahi khul rahi" jaisa dikhta hai kyunki process hi nahi bacha.
  Ab `uncaughtException`/`unhandledRejection` handlers hai jo error ko log mein print
  karte hai instead of process ko crash hone dena.
- **Port already-in-use / permission errors samajh nahi aate the**: raw stack trace
  ke bajaye ab clear message milega (e.g. "Port already in use" ya "no permission").
- **Startup par diagnostic banner add kiya**: server start hote hi PORT/HOST env
  values print hoti hai — hosting panel jo port/domain assign karta hai, wahi
  values yaha match honi chahiye. Mismatch hi sabse common reason hai "address khola
  to kuch nahi khula" wale issue ka.
- **`/health` endpoint add kiya**: kai hosting panels app ko "up" maanne ke liye
  ek simple GET request bhejte hai — ab `/health` plain "ok" text return karta hai.

### Agar phir bhi website nahi khul rahi (hosting-specific checklist)

Code locally train + serve + chat teeno test kiye, sab kaam kar rahe hai — isliye agar
tumhare host pe nahi khul raha to ye cheezein check karo:

1. **Panel jo port/domain deta hai wahi `.env` ke `PORT`/`HOST` se match hona chahiye**
   — startup log mein print hone wali values dekho.
2. **`node_modules` upload mat karo** — apne host par `npm install` chalao (isi zip mein
   `node_modules` nahi hai jaan-bujh kar, kyunki dusri machine ka `node_modules` upload
   karna hi bahut saari hosting-panel problems ki wajah hota hai).
3. **Kya panel Discord-bot-only hai ya web server bhi support karta hai?** — kayi free
   Discord bot hosting panels sirf bot process chalate hai, arbitrary HTTP web server ke
   liye alag se "website/subdomain" feature enable karna padta hai.
4. **Start command sahi hai?** — panel ko `node index.js` ya `npm start` chalana chahiye,
   `node index.js train` nahi (wo ek baar train karke exit ho jaata hai, koi web server
   nahi).
5. **Panel ke logs check karo** — ab crash hone par bhi clear error print hoga (upar wale
   fixes ki wajah se), wahi bata dega asli problem kya hai.

## Realistic expectations

- Chhota model (~5-25M params) chhote dataset pe train karke aap real transformer
  seekhoge aur ek genuine — lekin limited — chatbot milega.
- Quality seedha dataset ke size/quality pe depend karti hai. `data/example.jsonl`
  sirf format dikhane ke liye hai, usse train karke koi useful chatbot nahi banega.
- GPU nahi hai to training slow rahegi — chhota `CONTEXT_LENGTH`/`BATCH_SIZE` rakho.
