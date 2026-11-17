<?php
// Feature-bank-02 render harness. Downloaded library bytes are never edited.
//
// Usage: php harness.php case.json
// case.json: {"library": str, "adapter": path, "source": dir, "seed": int,
//             "output_stem": path-without-extension, "tmp": dir, "options": {...}}
//
// The adapter file is included inside a function scope and must set
// $GLOBALS['BANK_RESULT'] = ['label' => ..., 'params' => [...], optional 'bytes', 'shown_text'].
// If 'bytes' is absent, whatever the library printed to stdout is the image.
// A shutdown hook finalises even when a library calls exit().

$BANK_CFG = json_decode(file_get_contents($argv[1]), true, 512, JSON_THROW_ON_ERROR);
$GLOBALS['BANK_RESULT'] = null;
$GLOBALS['BANK_WARNINGS'] = [];
set_error_handler(function ($severity, $message, $file, $line) {
    $key = substr($message, 0, 160) . ' @' . basename((string) $file) . ':' . $line;
    // keep at most 50 distinct messages (some libraries emit one per pixel); count the rest
    if (isset($GLOBALS['BANK_WARNINGS'][$key]) || count($GLOBALS['BANK_WARNINGS']) < 50) {
        $GLOBALS['BANK_WARNINGS'][$key] = ($GLOBALS['BANK_WARNINGS'][$key] ?? 0) + 1;
    } else {
        $GLOBALS['BANK_WARNINGS']['_other_warnings'] = ($GLOBALS['BANK_WARNINGS']['_other_warnings'] ?? 0) + 1;
    }
    return true;
});

ini_set('session.save_path', $BANK_CFG['tmp']);
ini_set('session.use_cookies', '0');
ini_set('session.cache_limiter', '');
ini_set('display_errors', '0');
$_SESSION = [];
$_SERVER += ['REQUEST_METHOD' => 'GET', 'HTTP_HOST' => 'localhost', 'REQUEST_URI' => '/', 'REMOTE_ADDR' => '127.0.0.1'];


// Library randomness: seed the legacy Mersenne Twister used by rand()/mt_rand()/shuffle()/array_rand().
// random_int()/random_bytes() are CSPRNG and cannot be seeded; adapters record when a library uses them.
mt_srand((int) $BANK_CFG['seed']);
srand((int) $BANK_CFG['seed']);

// Independent stream for sampling adapter-level parameters, so parameter choices do not
// consume or depend on the library's own random stream.
$GLOBALS['BANK_RNG'] = new Random\Randomizer(new Random\Engine\Mt19937(((int) $BANK_CFG['seed']) ^ 0x5bd1e995));

// Optional draw-call interception (label maps + per-class on/off control); see intercept.php.
// Case keys: "intercept": true, "disable": ["line", "pixel", "text", ...] (call classes to suppress).
// The source rewriter is always installed: CSPRNG calls (random_int, random_bytes, openssl
// random) are routed to the seeded mt_rand in every run so renders are replayable. GD calls
// are rewritten only when interception is requested.
$GLOBALS['BANK_INTERCEPT'] = !empty($BANK_CFG['intercept']);
require __DIR__ . '/intercept.php';
BankGD\State::$gdIntercept = $GLOBALS['BANK_INTERCEPT'];
foreach ($BANK_CFG['disable'] ?? [] as $cls) BankGD\State::$disabled[$cls] = true;
// Chain mode: call ids continue after the previous stages' ids; label seeds from their maps.
if (!empty($BANK_CFG['options']['id_offset'])) BankGD\State::$next = (int) $BANK_CFG['options']['id_offset'];
if (!empty($BANK_CFG['options']['chain_labels'])) BankGD\State::$chainStem = $BANK_CFG['options']['chain_labels'];
BankGD\SourceRewriter::install([$BANK_CFG['source'], __DIR__], [__FILE__, __DIR__ . '/intercept.php']);

function bank_pick(array $values) { return $values[$GLOBALS['BANK_RNG']->pickArrayKeys($values, 1)[0]]; }
function bank_int(int $lo, int $hi): int { return $GLOBALS['BANK_RNG']->getInt($lo, $hi); }
function bank_float(float $lo, float $hi): float { return $lo + ($hi - $lo) * $GLOBALS['BANK_RNG']->getFloat(0, 1); }
function bank_bool(float $p = 0.5): bool { return $GLOBALS['BANK_RNG']->getFloat(0, 1) < $p; }
function bank_text(int $length, string $alphabet): string {
    $out = '';
    for ($i = 0; $i < $length; $i++) $out .= $alphabet[bank_int(0, strlen($alphabet) - 1)];
    return $out;
}
/**
 * Controlled parameter. Adapters always draw the sampled value first (so the adapter RNG stream is
 * identical whether or not a value is overridden), then the case's "params" override, if present,
 * replaces it. Every controlled value is logged in the case metadata under "param_control".
 */
function bank_param(string $name, $sampled) {
    $over = $GLOBALS['BANK_CFG']['params'] ?? [];
    $has = array_key_exists($name, $over);
    $v = $has ? $over[$name] : $sampled;
    $GLOBALS['BANK_PARAM_LOG'][$name] = ['value' => $v, 'overridden' => $has];
    return $v;
}
/**
 * Font from font bank 02 (175 OFL fonts) for libraries that accept a font file. Caps-only /
 * small-caps fonts are excluded unless the text is upper-case/digits only ($caseless = true).
 * Overridable via params["font"] (manifest path relative to sourcefree_synth); logged as a param.
 */
function bank_font(bool $caseless = false): string {
    static $rows = null, $calls = 0;
    $root = dirname(__DIR__, 2);
    if ($rows === null) $rows = json_decode(file_get_contents($root . '/resources/font_bank_02/fonts.json'), true);
    // Font holdout (resources/font_bank_02/font_split.json): only the case's split is sampled.
    // Option "font_split": "train" (default) | "heldout" | "any".
    $split = $GLOBALS['BANK_CFG']['options']['font_split'] ?? 'train';
    $pool = array_values(array_filter($rows, fn($r) => ($caseless || empty($r['caps_only']))
        && ($split === 'any' || ($r['split'] ?? 'train') === $split)));
    $sampled = bank_pick($pool)['path'];
    // params["fonts"]: a list consumed in order by successive bank_font() calls (multi-font libraries'
    // font sets); params["font"]: one font for every call. The sample is always drawn first.
    $list = $GLOBALS['BANK_CFG']['params']['fonts'] ?? null;
    $k = $calls++;
    if (is_array($list) && $list) $sampled = $list[$k % count($list)];
    $pick = bank_param('font', $sampled);
    $GLOBALS['BANK_PARAM_LOG']['fonts_chosen']['value'][] = $pick;
    $GLOBALS['BANK_PARAM_LOG']['fonts_chosen']['overridden'] = is_array($list) || isset($GLOBALS['BANK_CFG']['params']['font']);
    return bank_font_path($pick);
}
function bank_font_path(string $rel): string {
    $root = dirname(__DIR__, 2);
    return realpath($root . '/' . str_replace('\\', '/', $rel)) ?: ($root . '/' . $rel);
}
/**
 * Per-character font override for libraries that choose a font per glyph (patched call sites).
 * $orig is the library's own choice, evaluated first so the random stream is unchanged.
 * params["slot_fonts"][$slot] (bank path relative to sourcefree_synth) replaces it when given.
 */
function bank_slot_font(int $slot, string $orig): string {
    $over = $GLOBALS['BANK_CFG']['params']['slot_fonts'] ?? null;
    if (is_array($over) && isset($over[$slot]) && $over[$slot]) {
        $GLOBALS['BANK_PARAM_LOG']['slot_fonts'] = ['value' => $over, 'overridden' => true];
        return bank_font_path($over[$slot]);
    }
    return $orig;
}
/**
 * Controlled answer text. Samples a string of length $min..$max over $alphabet (always drawn,
 * so the adapter RNG stream is identical with or without an override), then applies params["text"]
 * if given. Overrides are case-folded when the library's alphabet is single-case; any character
 * still outside the alphabet raises (the case fails instead of drawing a wrong label).
 * The alphabet and length range are logged under param_control.
 */
function bank_label(string $alphabet, int $min = 2, int $max = 8, bool $unique = false): string {
    $length = bank_int($min, $max);
    if ($unique) {
        $chars = str_split($alphabet);
        shuffle($chars);
        $sampled = implode('', array_slice($chars, 0, $length));
    } else {
        $sampled = bank_text($length, $alphabet);
    }
    $t = bank_param('text', $sampled);
    $hasLower = (bool) preg_match('/[a-z]/', $alphabet);
    $hasUpper = (bool) preg_match('/[A-Z]/', $alphabet);
    if ($hasUpper && !$hasLower) $t = strtoupper($t);
    if ($hasLower && !$hasUpper) $t = strtolower($t);
    if (strspn($t, $alphabet) !== strlen($t)) throw new InvalidArgumentException("text '$t' outside library alphabet");
    if ($unique && count(array_unique(str_split($t))) !== strlen($t)) throw new InvalidArgumentException("text '$t' repeats a character this library cannot render twice");
    $GLOBALS['BANK_PARAM_LOG']['text']['value'] = $t;
    $GLOBALS['BANK_PARAM_LOG']['alphabet'] = ['value' => $alphabet, 'overridden' => false];
    $GLOBALS['BANK_PARAM_LOG']['length_range'] = ['value' => [$min, $max], 'overridden' => false];
    return $t;
}
/**
 * Chain mode (mix-and-match): option "bg_image" is the previous stage's final PNG. Adapters of
 * libraries that accept a background image pass bank_bg() to the library's own background input.
 * Option "chain_labels" is that stage's label-map stem: when the library loads the file, the
 * interceptor seeds its label canvases from those maps instead of one "loaded" id.
 */
function bank_bg(): ?string {
    $p = $GLOBALS['BANK_CFG']['options']['bg_image'] ?? null;
    if (!$p) return null;
    $p = realpath($p) ?: $p;
    BankGD\chain_register($p);
    return $p;
}
/** JPEG copy of the chain background (quality 95) for libraries that only read .jpg files. */
function bank_bg_jpeg(string $dir): ?string {
    $p = bank_bg();
    if (!$p) return null;
    @mkdir($dir, 0777, true);
    $j = rtrim($dir, '/\\') . '/bank_chain_bg.jpg';
    $im = \imagecreatefrompng($p);
    \imagejpeg($im, $j, 95);
    $j = realpath($j) ?: $j;
    BankGD\chain_register($j);
    return $j;
}
/** Relative path from directory $from to file $to (for libraries that only accept paths under their own dirs). */
function bank_relpath(string $from, string $to): string {
    $a = explode('/', trim(str_replace('\\', '/', realpath($from)), '/'));
    $b = explode('/', trim(str_replace('\\', '/', realpath($to)), '/'));
    $i = 0;
    while ($i < count($a) && $i < count($b) && strcasecmp($a[$i], $b[$i]) === 0) $i++;
    return str_repeat('../', count($a) - $i) . implode('/', array_slice($b, $i));
}
function bank_src(string $relative): string {
    $root = $GLOBALS['BANK_CFG']['source'];
    $hits = glob($root . '/*/' . $relative) ?: glob($root . '/' . $relative);
    if (!$hits) throw new RuntimeException("missing source file: $relative");
    return $hits[0];
}
function bank_root(): string {
    $root = $GLOBALS['BANK_CFG']['source'];
    $dirs = glob($root . '/*', GLOB_ONLYDIR);
    return count($dirs) === 1 ? $dirs[0] : $root;
}

// PHP-8 compatibility shim for unmaintained sources: apply exact, once-only textual
// replacements in memory and evaluate the result. Files on disk stay byte-identical;
// each patch is recorded in the case metadata. __DIR__/__FILE__ are rewritten to the
// original location so relative resource lookups keep working.
function bank_require_patched(string $file, array $replacements, bool $globals = false): void {
    $code = file_get_contents($file);
    foreach ($replacements as $from => $to) {
        if (str_starts_with($from, 're:')) {  // regex patch: 're:<pattern>' => replacement; must match at least once
            $code = preg_replace(substr($from, 3), $to, $code, -1, $n);
            if (!$n) throw new RuntimeException('regex patch matched nothing in ' . basename($file) . ': ' . $from);
            continue;
        }
        if (substr_count($code, $from) !== 1) throw new RuntimeException('patch target not unique in ' . basename($file) . ': ' . substr($from, 0, 60));
        $code = str_replace($from, $to, $code);
    }
    $code = str_replace(['__DIR__', '__FILE__'], [var_export(dirname($file), true), var_export($file, true)], $code);
    if ($globals) {
        // Top-level scripts whose functions use `global`: bind every script variable to global scope.
        preg_match_all('/\$([A-Za-z_][A-Za-z0-9_]*)/', $code, $m);
        $names = array_diff(array_unique($m[1]), ['this', 'GLOBALS', '_SESSION', '_SERVER', '_GET', '_POST', '_COOKIE', '_FILES', '_REQUEST', '_ENV']);
        $code = '<?php global $' . implode(', $', $names) . '; ?>' . $code;
    }
    $GLOBALS['BANK_PATCHES'][] = ['file' => basename($file), 'sha256_on_disk' => hash_file('sha256', $file), 'replacements' => $replacements];
    $code = BankGD\rewrite($code);  // CSPRNG always; GD calls only when intercepting
    eval('?>' . $code);
}

function bank_finalise(): void {
    static $done = false;
    if ($done) return;
    $done = true;
    $cfg = $GLOBALS['BANK_CFG'];
    $printed = '';
    while (ob_get_level() > 0) $printed = ob_get_clean() . $printed;
    $result = $GLOBALS['BANK_RESULT'] ?? $GLOBALS['BANK_PENDING'] ?? [];
    if (!isset($result['label']) && isset($result['label_session'])) {
        $v = $_SESSION;
        foreach ($result['label_session'] as $k) $v = is_array($v) ? ($v[$k] ?? null) : null;
        $result['label'] = $v;
    }
    $error = error_get_last();
    $bytes = $result['bytes'] ?? $printed;
    if (is_string($bytes) && preg_match('#^data:image/[a-z+]+;base64,#', $bytes)) {
        $bytes = base64_decode(substr($bytes, strpos($bytes, ',') + 1));
    }
    $meta = [
        'library' => $cfg['library'],
        'seed' => $cfg['seed'],
        'label' => $result['label'] ?? null,
        'shown_text' => $result['shown_text'] ?? ($result['label'] ?? null),
        'text_kind' => $result['text_kind'] ?? 'string',
        'label_case' => $result['label_case'] ?? 'exact',
        'params' => $result['params'] ?? [],
        'randomness' => $result['randomness'] ?? 'mt_rand seeded',
        'warnings' => $GLOBALS['BANK_WARNINGS'],
        'fatal' => ($error && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR], true)) ? $error['message'] : null,
        'exception' => $GLOBALS['BANK_EXCEPTION'] ?? null,
        'source_patches' => $GLOBALS['BANK_PATCHES'] ?? [],
        'param_control' => $GLOBALS['BANK_PARAM_LOG'] ?? [],
        'csprng_seeded' => true,
    ];
    $format = null;
    if (is_string($bytes) && $bytes !== '') {
        if (str_starts_with($bytes, "\x89PNG")) $format = 'png';
        elseif (str_starts_with($bytes, "\xFF\xD8")) $format = 'jpeg';
        elseif (str_starts_with($bytes, 'GIF8')) $format = 'gif';
        elseif (preg_match('#^\s*<#', $bytes) && stripos(substr($bytes, 0, 1000), '<svg') !== false) $format = 'svg';
    }
    $meta['source_format'] = $format;
    if ($format === 'svg') {
        file_put_contents($cfg['output_stem'] . '.svg', $bytes);
    } elseif ($format !== null) {
        file_put_contents($cfg['output_stem'] . '.' . $format, $bytes);
        if ($format === 'gif') $meta['gif_frames'] = substr_count($bytes, "\x21\xF9\x04");
        $im = @imagecreatefromstring($bytes);
        if ($im) {
            if (!imageistruecolor($im)) imagepalettetotruecolor($im);
            imagepng($im, $cfg['output_stem'] . '.png');
            $meta['width'] = imagesx($im);
            $meta['height'] = imagesy($im);
        }
    } else {
        $meta['stdout_head'] = is_string($bytes) ? substr($bytes, 0, 200) : gettype($bytes);
    }
    if ($GLOBALS['BANK_INTERCEPT'] && $format !== null && $format !== 'svg') {
        $meta['labels'] = BankGD\save($cfg['output_stem'], $meta['width'] ?? null, $meta['height'] ?? null);
    }
    $meta['ok'] = $format !== null && $meta['label'] !== null && $meta['label'] !== '' && $meta['fatal'] === null && $meta['exception'] === null;
    file_put_contents($cfg['output_stem'] . '.json', json_encode($meta, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_INVALID_UTF8_SUBSTITUTE));
}
register_shutdown_function('bank_finalise');

ob_start();
try {
    (function () { require $GLOBALS['BANK_CFG']['adapter']; })();
} catch (Throwable $e) {
    $GLOBALS['BANK_EXCEPTION'] = get_class($e) . ': ' . $e->getMessage() . ' @' . basename($e->getFile()) . ':' . $e->getLine();
}
bank_finalise();
