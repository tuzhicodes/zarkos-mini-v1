const path = require('path');
const config = require('./config');
const tf = require('./tf');

const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const C_PURPLE = '\x1b[38;5;135m';
const C_BLUE = '\x1b[38;5;33m';
const C_GREEN = '\x1b[92m';
const C_WHITE = '\x1b[97m';

const purple = (s) => `${C_PURPLE}${s}${RESET}`;
const blue = (s) => `${C_BLUE}${s}${RESET}`;
const green = (s) => `${C_GREEN}${s}${RESET}`;
const white = (s) => `${C_WHITE}${s}${RESET}`;
const boldGreen = (s) => `${BOLD}${C_GREEN}${s}${RESET}`;

function displayTitle() {
  return (config.modelName || 'ZarKos').toUpperCase().split('').join(' ');
}

function shortName() {
  return (config.modelName || 'ZarKos').split(' ')[0];
}

function printPreamble(commandName) {
  console.log(purple(`> ${config.modelName || 'ZarKos'} ${commandName}`));
  console.log(purple(`> node index.js ${commandName}`));
  const { device, backend, gpuConfirmed } = tf.deviceInfo();
  const gpuNote = device === 'gpu'
    ? (gpuConfirmed === true ? ', gpu confirmed' : gpuConfirmed === false ? ', gpu NOT confirmed — check nvidia-smi' : ', gpu unconfirmed — see note above')
    : '';
  printStatus(`device=${device}, backend=${backend}${gpuNote}`);
}

function printStatus(text) {
  console.log(purple(`> ${text}`));
}

function segLen(segments) {
  return segments.reduce((sum, [text]) => sum + text.length, 0);
}

function renderSegments(segments) {
  return segments.map(([text, colorFn]) => (colorFn ? colorFn(text) : text)).join('');
}

function boxTop(innerWidth) {
  return blue('╭' + '─'.repeat(innerWidth) + '╮');
}

function boxBottom(innerWidth) {
  return blue('╰' + '─'.repeat(innerWidth) + '╯');
}

function boxLine(segments, innerWidth, { align = 'left', indent = 0 } = {}) {
  const plainLen = segLen(segments);
  const totalPad = Math.max(0, innerWidth - plainLen);
  let leftPad, rightPad;
  if (align === 'center') {
    leftPad = Math.floor(totalPad / 2);
    rightPad = totalPad - leftPad;
  } else {
    leftPad = Math.min(indent, totalPad);
    rightPad = totalPad - leftPad;
  }
  return blue('│') + ' '.repeat(leftPad) + renderSegments(segments) + ' '.repeat(rightPad) + blue('│');
}

function printBanner(footerText) {
  const title = displayTitle();
  const madeBy = [];
  if (config.modelOwner) {
    madeBy.push([`MADE BY ${config.modelOwner.toUpperCase()}`, purple]);
    madeBy.push([' │ ', blue]);
  }
  madeBy.push([config.modelId || 'tuzhi/zarkos-mini-v1', white]);

  const footer = [[`${shortName()} ${footerText}`, boldGreen]];

  const contentWidths = [title.length, segLen(madeBy), segLen(footer)];
  const innerWidth = Math.max(...contentWidths) + 4;

  console.log(boxTop(innerWidth));
  console.log(boxLine([], innerWidth));
  console.log(boxLine([[title, purple]], innerWidth, { align: 'center' }));
  console.log(boxLine([], innerWidth));
  console.log(boxLine(madeBy, innerWidth, { align: 'center' }));
  console.log(boxBottom(innerWidth));
  console.log('');
  console.log(boxTop(innerWidth));
  console.log(boxLine(footer, innerWidth, { align: 'center' }));
  console.log(boxBottom(innerWidth));
  console.log('');
}

function printStartup(commandName, footerText, { status } = {}) {
  printPreamble(commandName);
  if (status) printStatus(status);
  printBanner(footerText);
}

function relativePath(absPath) {
  const rel = path.relative(process.cwd(), absPath);
  return rel === '' ? absPath : rel;
}

module.exports = {
  purple,
  blue,
  green,
  white,
  boldGreen,
  shortName,
  printPreamble,
  printStatus,
  printBanner,
  printStartup,
  relativePath,
};
