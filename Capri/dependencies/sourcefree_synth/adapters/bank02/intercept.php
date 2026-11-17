<?php
// Draw-call interception for feature bank 02 (hybrid v3 spec §2, §5).
//
// Library source is rewritten *in memory* as it is included (stream wrapper on file://),
// so every GD call goes through \BankGD\<fn>(). Each wrapper calls the real GD function with
// the identical arguments (output pixels unchanged) and mirrors the operation onto a hidden
// label canvas attached to the same image:
//   opaque[x,y]      id of the topmost opaque drawing call covering the pixel (0 = never drawn)
//   translucent[x,y] id of the topmost mostly-transparent call (alpha >= 64 of 127), if any
// Ids index the call log (function, call site, text, colour, ...). Text calls get one id per
// character. Pixel-moving operations are mirrored on the label canvases: imagecopy* family
// (honouring the source's transparent colour and merge percentage), imagerotate/scale/crop/
// flip, flood fills (pixel diff) and read-pixel/write-pixel warp loops (imagecolorat followed
// by imagesetpixel: the written pixel takes the label of the read whose colour is closest).
// Colour-only operations (imagefilter, imageconvolution) leave labels unchanged; geometric
// ones that cannot be mirrored (IMG_FILTER_SCATTER) are logged as 'unmirrored'.
// Nothing here calls mt_rand/rand, so library randomness is untouched.

namespace BankGD;

const MOSTLY_TRANSPARENT = 64;
// Per-class coverage bits (third label canvas): bit set wherever a call of that class ever drew,
// carried through copies/merges/warps. Unlike the id maps it survives occlusion and stacking.
const CLASS_BITS = ['line' => 1, 'rectangle' => 2, 'filled_rectangle' => 4, 'ellipse' => 8, 'filled_ellipse' => 16,
    'arc' => 32, 'filled_arc' => 64, 'polygon' => 128, 'filled_polygon' => 256, 'pixel' => 512, 'fill' => 1024,
    'text' => 2048, 'loaded' => 4096];

final class State
{
    public static \WeakMap $labels;       // GdImage => [opaque GdImage, translucent GdImage]
    public static \WeakMap $thickness;    // GdImage => int
    public static array $calls = [];      // id => record
    public static int $next = 1;
    public static array $reads = [];      // pending imagecolorat reads [img, x, y, colour]
    public static array $copySites = [];  // call sites that performed read->write copies
    public static ?array $run = null;     // current imagesetpixel run [site, id]
    public static ?\GdImage $emitted = null;
    public static ?\GdImage $lastTouched = null;
    public static array $disabled = [];   // call classes switched off by the controller
    public static bool $gdIntercept = false;  // false: only CSPRNG calls are rewritten
    public static ?array $glyphScope = null;  // ['char', 'index', 'id'] while a fontless glyph is drawn
    public static int $glyphIndex = 0;
    public static array $unmirrored = [];
    public static int $copyWrites = 0;
    public static int $suppressedPixels = 0;
    public static ?string $chainStem = null;  // previous chain stage's label-map stem
    public static array $chainPaths = [];     // files that are the previous stage's image
    public static int $chainSeeded = 0;
}
State::$labels = new \WeakMap();
State::$thickness = new \WeakMap();

// ---------------------------------------------------------------- label canvases

function blank(int $w, int $h): \GdImage
{
    $l = \imagecreatetruecolor($w, $h);
    \imagealphablending($l, false);
    \imageantialias($l, false);
    \imagefilledrectangle($l, 0, 0, $w - 1, $h - 1, 0);
    return $l;
}

function attach(\GdImage $im, int $fillId = 0): void
{
    $o = blank(\imagesx($im), \imagesy($im));
    $m = blank(\imagesx($im), \imagesy($im));
    if ($fillId) {
        \imagefilledrectangle($o, 0, 0, \imagesx($im) - 1, \imagesy($im) - 1, $fillId);
        \imagefilledrectangle($m, 0, 0, \imagesx($im) - 1, \imagesy($im) - 1, CLASS_BITS['loaded']);
    }
    State::$labels[$im] = [$o, blank(\imagesx($im), \imagesy($im)), $m];
}

function lab(\GdImage $im): array
{
    if (!isset(State::$labels[$im])) {
        // Chain mode: a background loaded through a dynamic call ($fn($path), e.g. Kohana) is not
        // rewritten; the first untracked image with the background's exact size is that file.
        if (!(State::$chainStem && State::$chainPaths && State::$chainSeeded === 0 && chain_seed($im))) attach($im);
    }
    return State::$labels[$im];
}

function site(int $unused = 0): string
{
    // first frame outside this file = the library's call into a wrapper
    $bt = \debug_backtrace(\DEBUG_BACKTRACE_IGNORE_ARGS, 12);
    foreach ($bt as $k => $f) {
        if (($f['file'] ?? __FILE__) !== __FILE__) {
            $caller = $bt[$k + 1] ?? [];
            $fn = ($caller['class'] ?? '') ? $caller['class'] . '::' . $caller['function'] : ($caller['function'] ?? '');
            return \basename($f['file']) . ':' . ($f['line'] ?? 0) . ' ' . $fn;
        }
    }
    return '?';
}

function rgba(\GdImage $im, int $colour): ?array
{
    if ($colour < 0) return null;  // IMG_COLOR_STYLED / BRUSHED / TILED etc.
    $c = @\imagecolorsforindex($im, $colour);
    return $c ? [$c['red'], $c['green'], $c['blue'], $c['alpha']] : null;
}

/**
 * Glyph scopes for libraries that draw characters without text calls (vector strokes, sprite
 * copies). Adapters insert glyph_begin($char) / glyph_end() around the per-character drawing
 * (in-memory source patch). Inside a scope every drawing call or copy is labelled as that
 * character (one id per glyph, class 'text', primitive kept), gets the text coverage bit, and
 * is switched off by disable('text'). No-ops when interception is off.
 */
function glyph_begin(string $char): void
{
    if (!State::$gdIntercept) return;
    State::$glyphScope = ['char' => $char, 'index' => State::$glyphIndex++, 'id' => null];
}
function glyph_end(): void { State::$glyphScope = null; }
function effClass(string $class): string { return State::$glyphScope ? 'text' : $class; }

function newId(string $fn, \GdImage $im, $colour, array $extra = []): int
{
    if (State::$glyphScope && ($extra['class'] ?? '') !== 'filter') {
        if (State::$glyphScope['id'] !== null) {
            State::$calls[State::$glyphScope['id']]['primitives'][] = $fn;
            State::$lastTouched = $im;
            return State::$glyphScope['id'];
        }
        $extra = ['class' => 'text', 'char' => State::$glyphScope['char'], 'char_index' => State::$glyphScope['index'],
                  'glyph_scope' => true, 'primitives' => [$fn]] + $extra;
        $extra['class'] = 'text';
    }
    $id = State::$next++;
    $c = \is_int($colour) ? rgba($im, $colour) : null;
    State::$calls[$id] = ['id' => $id, 'fn' => $fn, 'site' => site(3), 'rgba' => $c,
        'translucent' => $c !== null && $c[3] >= MOSTLY_TRANSPARENT] + $extra;
    if (State::$glyphScope && ($extra['glyph_scope'] ?? false)) {
        State::$glyphScope['id'] = $id;
        State::$calls[$id]['translucent'] = false;  // a glyph's strokes are its body, whatever their alpha
    }
    State::$run = null;
    State::$reads = [];   // a new drawing call ends any pending read->write copy
    State::$lastTouched = $im;
    return $id;
}

function target(\GdImage $im, int $id): \GdImage
{
    [$o, $t] = lab($im);
    return State::$calls[$id]['translucent'] ? $t : $o;
}

/** OR a class bit into the coverage canvas wherever label image $l now holds $id, inside the box. */
function markClass(\GdImage $im, \GdImage $l, int $id, string $class, array $box): void
{
    $bit = CLASS_BITS[$class] ?? 0;
    if (!$bit) return;
    $m = lab($im)[2];
    $W = \imagesx($l); $H = \imagesy($l);
    $x0 = \max(0, (int) \floor($box[0])); $y0 = \max(0, (int) \floor($box[1]));
    $x1 = \min($W - 1, (int) \ceil($box[2])); $y1 = \min($H - 1, (int) \ceil($box[3]));
    for ($y = $y0; $y <= $y1; $y++) for ($x = $x0; $x <= $x1; $x++) {
        if (\imagecolorat($l, $x, $y) === $id) \imagesetpixel($m, $x, $y, \imagecolorat($m, $x, $y) | $bit);
    }
}

function orBit(\GdImage $im, int $x, int $y, int $bit): void
{
    $m = lab($im)[2];
    if ($x >= 0 && $y >= 0 && $x < \imagesx($m) && $y < \imagesy($m)) \imagesetpixel($m, $x, $y, \imagecolorat($m, $x, $y) | $bit);
}

/** Bounding box of a shape call's arguments (padded by line thickness). */
function shapeBox(string $fn, array $a, int $t): array
{
    $p = $t + 2;
    switch ($fn) {
        case 'imageline': case 'imagedashedline': case 'imagerectangle': case 'imagefilledrectangle':
            return [\min($a[1], $a[3]) - $p, \min($a[2], $a[4]) - $p, \max($a[1], $a[3]) + $p, \max($a[2], $a[4]) + $p];
        case 'imageellipse': case 'imagefilledellipse': case 'imagearc': case 'imagefilledarc':
            $hw = \abs($a[3]) / 2; $hh = \abs($a[4]) / 2;  // GD accepts negative sizes
            return [$a[1] - $hw - $p, $a[2] - $hh - $p, $a[1] + $hw + $p, $a[2] + $hh + $p];
        default:  // polygons: points array
            $pts = $a[1]; $xs = []; $ys = [];
            for ($i = 0; $i + 1 < \count($pts); $i += 2) { $xs[] = $pts[$i]; $ys[] = $pts[$i + 1]; }
            return $xs ? [\min($xs) - $p, \min($ys) - $p, \max($xs) + $p, \max($ys) + $p] : [0, 0, PHP_INT_MAX, PHP_INT_MAX];
    }
}

function mirrorThickness(\GdImage $im, \GdImage $l): void
{
    \imagesetthickness($l, State::$thickness[$im] ?? 1);
}

/**
 * Switch check. Entries in State::$disabled are either a call class ("line", "pixel", "text", ...)
 * or "site:<substring>", which suppresses calls whose library call site (file:line function)
 * contains the substring, e.g. "site:drawLine" or "site:writeNoise".
 */
function disabled(string $class): bool
{
    if (isset(State::$disabled[$class])) return true;
    foreach (State::$disabled as $k => $_) {
        if (\str_starts_with($k, 'site:') && \str_contains(site(), \substr($k, 5))) return true;
    }
    return false;
}

/** imagettftext()'s return value (absolute corner points) computed without drawing. */
function ttfReturn($size, $angle, $font, $text, $x, $y): array
{
    $b = \imagettfbbox($size, $angle, $font, $text);
    for ($i = 0; $i < 8; $i += 2) { $b[$i] += (int) $x; $b[$i + 1] += (int) $y; }
    return $b;
}

// ---------------------------------------------------------------- creation / loading / output

function imagecreatetruecolor($w, $h) { $im = \imagecreatetruecolor($w, $h); if ($im) attach($im); return $im; }
function imagecreate($w, $h) { $im = \imagecreate($w, $h); if ($im) attach($im); return $im; }
function loaded($im, string $fn, $src)
{
    if ($im instanceof \GdImage) {
        $id = State::$next++;
        State::$calls[$id] = ['id' => $id, 'fn' => $fn, 'site' => site(2), 'source' => \is_string($src) ? \basename($src) : null,
            'rgba' => null, 'translucent' => false];
        $real = \is_string($src) ? \str_replace('\\', '/', (\realpath($src) ?: $src)) : null;
        if ($real !== null && isset(State::$chainPaths[\strtolower($real)]) && State::$chainStem && chain_seed($im)) {
            State::$calls[$id]['chain_bg'] = true;   // labels carried from the previous chain stage
        } else {
            attach($im, $id);
        }
    }
    return $im;
}

/** Chain mode: register a file the library will load as the previous stage's image. */
function chain_register(string $path): void { State::$chainPaths[\strtolower(\str_replace('\\', '/', $path))] = true; }

/** Seed an image's label canvases from the previous stage's saved maps (same pixel grid). */
function chain_seed(\GdImage $im): bool
{
    $stem = State::$chainStem;
    $maps = [];
    foreach (['.labels.png', '.labels_translucent.png', '.labels_classes.png'] as $suffix) {
        $l = @\imagecreatefrompng($stem . $suffix);
        if (!$l || \imagesx($l) !== \imagesx($im) || \imagesy($l) !== \imagesy($im)) return false;
        if (!\imageistruecolor($l)) \imagepalettetotruecolor($l);
        \imagealphablending($l, false);
        \imageantialias($l, false);
        $maps[] = $l;
    }
    State::$labels[$im] = $maps;
    State::$chainSeeded++;
    return true;
}
function imagecreatefrompng($f) { return loaded(\imagecreatefrompng($f), 'imagecreatefrompng', $f); }
function imagecreatefromjpeg($f) { return loaded(\imagecreatefromjpeg($f), 'imagecreatefromjpeg', $f); }
function imagecreatefromgif($f) { return loaded(\imagecreatefromgif($f), 'imagecreatefromgif', $f); }
function imagecreatefromwebp($f) { return loaded(\imagecreatefromwebp($f), 'imagecreatefromwebp', $f); }
function imagecreatefrombmp($f) { return loaded(\imagecreatefrombmp($f), 'imagecreatefrombmp', $f); }
function imagecreatefromstring($s) { return loaded(\imagecreatefromstring($s), 'imagecreatefromstring', null); }

function emitted($im) { if ($im instanceof \GdImage) State::$emitted = $im; }
function imagepng($im, ...$a) { emitted($im); return \imagepng($im, ...$a); }
function imagejpeg($im, ...$a) { emitted($im); return \imagejpeg($im, ...$a); }
function imagegif($im, ...$a) { emitted($im); return \imagegif($im, ...$a); }
function imagewebp($im, ...$a) { emitted($im); return \imagewebp($im, ...$a); }
function imagebmp($im, ...$a) { emitted($im); return \imagebmp($im, ...$a); }

function imagesetthickness($im, $t) { State::$thickness[$im] = (int) $t; return \imagesetthickness($im, $t); }
function imagepalettetotruecolor($im) { return \imagepalettetotruecolor($im); }

// ---------------------------------------------------------------- reads (for warp loops)

function imagecolorat($im, $x, $y)
{
    $c = \imagecolorat($im, $x, $y);
    if ($c !== false) {
        State::$reads[] = [$im, (int) $x, (int) $y, $c];
        if (\count(State::$reads) > 16) \array_shift(State::$reads);
    }
    return $c;
}

// ---------------------------------------------------------------- primitive drawing

function shape(string $fn, string $class, array $args)
{
    $im = $args[0];
    $colour = $args[\count($args) - 1];
    $class = effClass($class);
    if (disabled($class)) { newId($fn, $im, $colour, ['class' => $class, 'suppressed' => true]); return true; }
    $r = ('\\' . $fn)(...$args);
    $id = newId($fn, $im, $colour, ['class' => $class]);
    $l = target($im, $id);
    mirrorThickness($im, $l);
    $orig = $args;
    $args[0] = $l;
    $args[\count($args) - 1] = $id;
    ('\\' . $fn)(...$args);
    markClass($im, $l, $id, $class, shapeBox($fn, $orig, State::$thickness[$im] ?? 1));
    return $r;
}
function imageline(...$a) { return shape('imageline', 'line', $a); }
function imagedashedline(...$a) { return shape('imagedashedline', 'line', $a); }
function imagerectangle(...$a) { return shape('imagerectangle', 'rectangle', $a); }
function imagefilledrectangle(...$a) { return shape('imagefilledrectangle', 'filled_rectangle', $a); }
function imageellipse(...$a) { return shape('imageellipse', 'ellipse', $a); }
function imagefilledellipse(...$a) { return shape('imagefilledellipse', 'filled_ellipse', $a); }
function imagearc(...$a) { return shape('imagearc', 'arc', $a); }
function imagepolygon(...$a) { return shape('imagepolygon', 'polygon', $a); }
function imageopenpolygon(...$a) { return shape('imageopenpolygon', 'polygon', $a); }
function imagefilledpolygon(...$a) { return shape('imagefilledpolygon', 'filled_polygon', $a); }

function imagefilledarc($im, $cx, $cy, $w, $h, $s, $e, $colour, $style)
{
    if (disabled('filled_arc')) { newId('imagefilledarc', $im, $colour, ['class' => 'filled_arc', 'suppressed' => true]); return true; }
    $r = \imagefilledarc($im, $cx, $cy, $w, $h, $s, $e, $colour, $style);
    $id = newId('imagefilledarc', $im, $colour, ['class' => 'filled_arc']);
    $l = target($im, $id);
    \imagefilledarc($l, $cx, $cy, $w, $h, $s, $e, $id, $style);
    markClass($im, $l, $id, 'filled_arc', [$cx - $w / 2 - 2, $cy - $h / 2 - 2, $cx + $w / 2 + 2, $cy + $h / 2 + 2]);
    return $r;
}

function imagesetpixel($im, $x, $y, $colour)
{
    // Read->write copy (warp loop): label follows the read pixel with the closest colour.
    if (State::$reads) {
        $best = null; $bd = PHP_INT_MAX;
        foreach (State::$reads as $rd) {
            $d = coldist($rd[0], $rd[3], $im, $colour);
            if ($d < $bd) { $bd = $d; $best = $rd; }
        }
        State::$reads = [];
        $r = \imagesetpixel($im, $x, $y, $colour);
        [$so, $st] = lab($best[0]);
        [$do, $dt] = lab($im);
        if ($best[1] >= 0 && $best[2] >= 0 && $best[1] < \imagesx($so) && $best[2] < \imagesy($so)) {
            \imagesetpixel($do, $x, $y, \imagecolorat($so, $best[1], $best[2]));
            \imagesetpixel($dt, $x, $y, \imagecolorat($st, $best[1], $best[2]));
            \imagesetpixel(lab($im)[2], $x, $y, \imagecolorat(lab($best[0])[2], $best[1], $best[2]));
        }
        State::$copySites[site(1)] = true;
        State::$copyWrites++;
        State::$lastTouched = $im;
        return $r;
    }
    $s = site(1);
    if (isset(State::$copySites[$s])) {  // copy loop writing background with no source read
        [$do, $dt] = lab($im);
        \imagesetpixel($do, $x, $y, 0); \imagesetpixel($dt, $x, $y, 0); \imagesetpixel(lab($im)[2], $x, $y, 0);
        return \imagesetpixel($im, $x, $y, $colour);
    }
    if (disabled('pixel')) { State::$suppressedPixels++; return true; }
    $r = \imagesetpixel($im, $x, $y, $colour);
    $c = rgba($im, $colour);
    $translucent = $c !== null && $c[3] >= MOSTLY_TRANSPARENT;
    if (!State::$run || State::$run[0] !== $s || State::$run[2] !== $translucent) {
        $id = newId('imagesetpixel', $im, $colour, ['class' => 'pixel', 'n' => 0]);
        State::$run = [$s, $id, $translucent];
    }
    $id = State::$run[1];
    State::$calls[$id]['n']++;
    \imagesetpixel(target($im, $id), $x, $y, $id);
    orBit($im, (int) $x, (int) $y, CLASS_BITS['pixel']);
    State::$lastTouched = $im;
    return $r;
}

function coldist(\GdImage $a, int $ca, \GdImage $b, int $cb): int
{
    $x = @\imagecolorsforindex($a, $ca); $y = @\imagecolorsforindex($b, $cb);
    if (!$x || !$y) return PHP_INT_MAX;
    return \abs($x['red'] - $y['red']) + \abs($x['green'] - $y['green']) + \abs($x['blue'] - $y['blue']);
}

// ---------------------------------------------------------------- flood fills (pixel diff)

function snapshot(\GdImage $im): array
{
    $w = \imagesx($im); $h = \imagesy($im); $p = [];
    for ($y = 0; $y < $h; $y++) for ($x = 0; $x < $w; $x++) $p[] = \imagecolorat($im, $x, $y);
    return $p;
}
function diffLabel(\GdImage $im, array $before, int $id): int
{
    $w = \imagesx($im); $h = \imagesy($im); $l = target($im, $id); $i = 0; $n = 0;
    for ($y = 0; $y < $h; $y++) for ($x = 0; $x < $w; $x++, $i++) {
        if (\imagecolorat($im, $x, $y) !== $before[$i]) { \imagesetpixel($l, $x, $y, $id); orBit($im, $x, $y, CLASS_BITS['fill']); $n++; }
    }
    return $n;
}
function fill(string $fn, array $args)
{
    $im = $args[0];
    $colour = $args[\count($args) - 1];
    if (disabled('fill')) { newId($fn, $im, $colour, ['class' => 'fill', 'suppressed' => true]); return true; }
    $before = snapshot($im);
    $r = ('\\' . $fn)(...$args);
    $id = newId($fn, $im, $colour, ['class' => 'fill']);
    State::$calls[$id]['pixels'] = diffLabel($im, $before, $id);
    State::$calls[$id]['canvas_fraction'] = State::$calls[$id]['pixels'] / (\imagesx($im) * \imagesy($im));
    return $r;
}
function imagefill(...$a) { return fill('imagefill', $a); }
function imagefilltoborder(...$a) { return fill('imagefilltoborder', $a); }

// ---------------------------------------------------------------- text (one id per character)

/** Characters as GD sees them: UTF-8 when valid, else one byte each (GD's Latin-1 fallback). */
function chars(string $text): array
{
    if ($text === '') return [];
    return \mb_check_encoding($text, 'UTF-8') ? \mb_str_split($text) : \str_split($text);
}

function charIds(\GdImage $im, string $fn, $colour, string $text, array $extra): array
{
    $ids = [];
    foreach (chars($text) as $i => $ch) {
        $ids[] = newId($fn, $im, $colour, ['class' => 'text', 'char' => $ch, 'char_index' => $i, 'text' => $text] + $extra);
    }
    return $ids;
}

function ttf(string $fn, $im, $size, $angle, $x, $y, $colour, $font, $text, array $opt = [])
{
    if (disabled('text')) { newId($fn, $im, $colour, ['class' => 'text', 'text' => (string) $text, 'suppressed' => true]); return ttfReturn($size, $angle, $font, $text, $x, $y); }
    $r = $fn === 'imagefttext' ? \imagefttext($im, $size, $angle, $x, $y, $colour, $font, $text, $opt)
                               : \imagettftext($im, $size, $angle, $x, $y, $colour, $font, $text);
    $fp = (string) $font;
    $ids = charIds($im, $fn, $colour, (string) $text, ['size' => $size, 'angle' => $angle, 'font' => \basename($fp),
        'font_path' => \str_replace('\\', '/', (\realpath($fp) ?: $fp))]);
    $rad = \deg2rad((float) $angle);
    $chars = chars((string) $text);
    foreach ($chars as $i => $ch) {
        $adv = 0.0;
        if ($i > 0) {
            // pen position of char i (includes kerning; side bearings cancel)
            $b = \imagettfbbox($size, 0, $font, \implode('', \array_slice($chars, 0, $i)) . $ch);
            $c = \imagettfbbox($size, 0, $font, $ch);
            $adv = $b[2] - $c[2];
        }
        // GD truncates float coordinates to int; the pen advance is added on top
        $cx = (int) $x + (int) \round($adv * \cos($rad));
        $cy = (int) $y - (int) \round($adv * \sin($rad));
        aaGlyph($im, target($im, $ids[$i]), $ids[$i], $size, $angle, $cx, $cy, $font, $ch);
    }
    return $r;
}

/**
 * Label one TTF glyph from GD's own antialiased raster (same hinting as the real draw; the
 * non-antialiased raster is hinted differently and can sit a pixel off on thin fonts).
 * Id map: pixels with >= 50% coverage. Coverage bit: every pixel the glyph touches at all.
 */
function aaGlyph(\GdImage $im, \GdImage $l, int $id, $size, $angle, int $cx, int $cy, $font, string $ch): void
{
    $b = @\imagettfbbox($size, $angle, $font, $ch);
    if (!$b) return;
    $xs = [$b[0], $b[2], $b[4], $b[6]]; $ys = [$b[1], $b[3], $b[5], $b[7]];
    $x0 = \min($xs) + $cx - 4; $x1 = \max($xs) + $cx + 4; $y0 = \min($ys) + $cy - 4; $y1 = \max($ys) + $cy + 4;
    $W = \imagesx($l); $H = \imagesy($l);
    $s = \imagecreatetruecolor($x1 - $x0 + 1, $y1 - $y0 + 1);  // black, blending on
    \imagettftext($s, $size, $angle, $cx - $x0, $cy - $y0, 0xFFFFFF, $font, $ch);
    $m = lab($im)[2];
    $bit = CLASS_BITS['text'];
    // raster size of the glyph (>= 50% coverage), in total and inside the canvas: clipping evidence
    $all = 0; $in = 0;
    for ($y = $y0; $y <= $y1; $y++) {
        for ($x = $x0; $x <= $x1; $x++) {
            $v = \imagecolorat($s, $x - $x0, $y - $y0) & 0xFF;
            if ($v === 0) continue;
            $inside = $x >= 0 && $y >= 0 && $x < $W && $y < $H;
            if ($v >= 128) { $all++; if ($inside) $in++; }
            if (!$inside) continue;
            if ($v >= 128) \imagesetpixel($l, $x, $y, $id);
            \imagesetpixel($m, $x, $y, \imagecolorat($m, $x, $y) | $bit);
        }
    }
    State::$calls[$id]['raster_px'] = $all;
    State::$calls[$id]['raster_px_in'] = $in;
}
function imagettftext($im, $size, $angle, $x, $y, $colour, $font, $text) { return ttf('imagettftext', $im, $size, $angle, $x, $y, $colour, $font, $text); }
function imagefttext($im, $size, $angle, $x, $y, $colour, $font, $text, $opt = []) { return ttf('imagefttext', $im, $size, $angle, $x, $y, $colour, $font, $text, $opt); }

function bitmap(string $fn, $im, $font, $x, $y, $text, $colour)
{
    if (disabled('text')) { newId($fn, $im, $colour, ['class' => 'text', 'text' => (string) $text, 'suppressed' => true]); return true; }
    $r = ('\\' . $fn)($im, $font, $x, $y, $text, $colour);
    $text = (string) $text;
    if ($fn === 'imagechar' || $fn === 'imagecharup') $text = \substr($text, 0, 1);
    $ids = charIds($im, $fn, $colour, $text, ['bitmap_font' => \is_int($font) ? $font : 'loaded']);
    $fw = \imagefontwidth($font);
    foreach (\str_split($text) as $i => $ch) {
        $up = \str_ends_with($fn, 'up');
        $cx = $up ? $x : $x + $i * $fw;
        $cy = $up ? $y - $i * $fw : $y;
        $l = target($im, $ids[$i]);
        ('\\' . $fn)($l, $font, $cx, $cy, $ch, $ids[$i]);
        $fh = \imagefontheight($font);
        $box = $up ? [$cx - 1, $cy - $fw - 1, $cx + $fh + 1, $cy + 1] : [$cx - 1, $cy - 1, $cx + $fw + 1, $cy + $fh + 1];
        markClass($im, $l, $ids[$i], 'text', $box);
        // raster size of the glyph, in total (scratch draw) and inside the canvas (clipping evidence)
        $sw = ($up ? $fh : $fw) + 4; $sh = ($up ? $fw : $fh) + 4;
        $s = \imagecreatetruecolor($sw, $sh);
        ('\\' . $fn)($s, $font, 2, $up ? $sh - 2 : 2, $ch, 0xFFFFFF);
        $all = 0;
        for ($y = 0; $y < $sh; $y++) for ($x = 0; $x < $sw; $x++) if (\imagecolorat($s, $x, $y)) $all++;
        $in = 0;
        for ($y = \max(0, (int) $box[1]); $y <= \min(\imagesy($l) - 1, (int) $box[3]); $y++)
            for ($x = \max(0, (int) $box[0]); $x <= \min(\imagesx($l) - 1, (int) $box[2]); $x++)
                if (\imagecolorat($l, $x, $y) === $ids[$i]) $in++;
        State::$calls[$ids[$i]]['raster_px'] = $all;
        State::$calls[$ids[$i]]['raster_px_in'] = \min($in, $all);
    }
    return $r;
}
function imagestring($im, $font, $x, $y, $s, $c) { return bitmap('imagestring', $im, $font, $x, $y, $s, $c); }
function imagestringup($im, $font, $x, $y, $s, $c) { return bitmap('imagestringup', $im, $font, $x, $y, $s, $c); }
function imagechar($im, $font, $x, $y, $s, $c) { return bitmap('imagechar', $im, $font, $x, $y, $s, $c); }
function imagecharup($im, $font, $x, $y, $s, $c) { return bitmap('imagecharup', $im, $font, $x, $y, $s, $c); }

// ---------------------------------------------------------------- copies

function copyLabels(\GdImage $dst, \GdImage $src, int $dx, int $dy, int $sx, int $sy, int $dw, int $dh, int $sw, int $sh, int $pct): void
{
    $scopeId = State::$glyphScope ? newId('imagecopy', $dst, null, ['class' => 'text']) : 0;
    [$so, $st, $sm] = lab($src);
    [$do, $dt, $dm] = lab($dst);
    $trans = \imagecolortransparent($src);
    $srcTrue = \imageistruecolor($src);
    $W = \imagesx($dst); $H = \imagesy($dst); $SW = \imagesx($src); $SH = \imagesy($src);
    for ($j = 0; $j < $dh; $j++) {
        $y = $dy + $j; if ($y < 0 || $y >= $H) continue;
        $syy = $sy + (int) \floor($j * $sh / $dh); if ($syy < 0 || $syy >= $SH) continue;
        for ($i = 0; $i < $dw; $i++) {
            $x = $dx + $i; if ($x < 0 || $x >= $W) continue;
            $sxx = $sx + (int) \floor($i * $sw / $dw); if ($sxx < 0 || $sxx >= $SW) continue;
            $real = \imagecolorat($src, $sxx, $syy);
            if ($trans >= 0 && $real === $trans && !$srcTrue) continue;          // palette transparency skipped by GD
            if ($srcTrue && (($real >> 24) & 0x7F) === 127) continue;            // fully transparent truecolor pixel
            $o = \imagecolorat($so, $sxx, $syy);
            $t = \imagecolorat($st, $sxx, $syy);
            if ($scopeId) { $o = $scopeId; $t = 0; }
            $mk = $scopeId ? CLASS_BITS['text'] : \imagecolorat($sm, $sxx, $syy);
            if ($pct >= 100) {                       // plain copy: source replaces destination
                \imagesetpixel($do, $x, $y, $o);
                if ($t) \imagesetpixel($dt, $x, $y, $t);
                \imagesetpixel($dm, $x, $y, $mk);
                continue;
            }
            // Partial merge: both layers stay visible in the blend. Coverage bits are combined; the
            // dominant side (by merge percentage) takes the opaque id map, the other the translucent one.
            if ($mk) \imagesetpixel($dm, $x, $y, \imagecolorat($dm, $x, $y) | $mk);
            if ($o === 0) { if ($t) \imagesetpixel($dt, $x, $y, $t); continue; }
            $od = \imagecolorat($do, $x, $y);
            if ($pct >= 50) { \imagesetpixel($do, $x, $y, $o); if ($od) \imagesetpixel($dt, $x, $y, $od); }
            else { \imagesetpixel($dt, $x, $y, $o); }
        }
    }
}
function imagecopy($d, $s, $dx, $dy, $sx, $sy, $w, $h) { if (State::$glyphScope && disabled('text')) { newId('imagecopy', $d, null, ['class' => 'text', 'suppressed' => true]); return true; } $r = \imagecopy($d, $s, $dx, $dy, $sx, $sy, $w, $h); copyLabels($d, $s, $dx, $dy, $sx, $sy, $w, $h, $w, $h, 100); State::$lastTouched = $d; return $r; }
function imagecopymerge($d, $s, $dx, $dy, $sx, $sy, $w, $h, $p) { $r = \imagecopymerge($d, $s, $dx, $dy, $sx, $sy, $w, $h, $p); copyLabels($d, $s, $dx, $dy, $sx, $sy, $w, $h, $w, $h, (int) $p); State::$lastTouched = $d; return $r; }
function imagecopymergegray($d, $s, $dx, $dy, $sx, $sy, $w, $h, $p) { $r = \imagecopymergegray($d, $s, $dx, $dy, $sx, $sy, $w, $h, $p); copyLabels($d, $s, $dx, $dy, $sx, $sy, $w, $h, $w, $h, (int) $p); State::$lastTouched = $d; return $r; }
function imagecopyresized($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh) { $r = \imagecopyresized($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh); copyLabels($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh, 100); State::$lastTouched = $d; return $r; }
function imagecopyresampled($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh) { $r = \imagecopyresampled($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh); copyLabels($d, $s, $dx, $dy, $sx, $sy, $dw, $dh, $sw, $sh, 100); State::$lastTouched = $d; return $r; }

// ---------------------------------------------------------------- whole-image geometry

function geometric(\GdImage $src, $out, callable $op): mixed
{
    if (!($out instanceof \GdImage)) return $out;
    $ls = [];
    foreach (lab($src) as $l) {
        \imagesetinterpolation($l, \IMG_NEAREST_NEIGHBOUR);
        $r = $op($l);
        if ($r instanceof \GdImage) { \imagealphablending($r, false); $ls[] = $r; }
        else $ls[] = blank(\imagesx($out), \imagesy($out));
    }
    State::$labels[$out] = $ls;
    State::$lastTouched = $out;
    return $out;
}
function imagerotate($im, $angle, $bg, ...$rest) { return geometric($im, \imagerotate($im, $angle, $bg, ...$rest), fn($l) => \imagerotate($l, $angle, 0)); }
function imagescale($im, $w, $h = -1, $mode = \IMG_BILINEAR_FIXED) { return geometric($im, \imagescale($im, $w, $h, $mode), fn($l) => \imagescale($l, $w, $h, \IMG_NEAREST_NEIGHBOUR)); }
function imagecrop($im, $rect) { return geometric($im, \imagecrop($im, $rect), fn($l) => \imagecrop($l, $rect)); }
function imageflip($im, $mode) { $r = \imageflip($im, $mode); foreach (lab($im) as $l) \imageflip($l, $mode); return $r; }

function imagefilter($im, $filter, ...$args)
{
    $r = \imagefilter($im, $filter, ...$args);
    if ($filter === \IMG_FILTER_SCATTER || $filter === \IMG_FILTER_PIXELATE) State::$unmirrored[] = ['imagefilter', $filter, site(1)];
    State::$calls[State::$next] = ['id' => State::$next, 'fn' => 'imagefilter', 'filter' => $filter, 'args' => $args, 'site' => site(1), 'class' => 'filter', 'rgba' => null, 'translucent' => false];
    State::$next++;
    return $r;
}
function imageconvolution($im, $m, $d, $o)
{
    $r = \imageconvolution($im, $m, $d, $o);
    State::$calls[State::$next] = ['id' => State::$next, 'fn' => 'imageconvolution', 'site' => site(1), 'class' => 'filter', 'rgba' => null, 'translucent' => false];
    State::$next++;
    return $r;
}

// ---------------------------------------------------------------- source rewriting

const WRAPPED = ['imagecreatetruecolor', 'imagecreate', 'imagecreatefrompng', 'imagecreatefromjpeg', 'imagecreatefromgif',
    'imagecreatefromwebp', 'imagecreatefrombmp', 'imagecreatefromstring', 'imagepng', 'imagejpeg', 'imagegif', 'imagewebp',
    'imagebmp', 'imagesetthickness', 'imagecolorat', 'imageline', 'imagedashedline', 'imagerectangle', 'imagefilledrectangle',
    'imageellipse', 'imagefilledellipse', 'imagearc', 'imagefilledarc', 'imagepolygon', 'imageopenpolygon', 'imagefilledpolygon',
    'imagesetpixel', 'imagefill', 'imagefilltoborder', 'imagettftext', 'imagefttext', 'imagestring', 'imagestringup',
    'imagechar', 'imagecharup', 'imagecopy', 'imagecopymerge', 'imagecopymergegray', 'imagecopyresized', 'imagecopyresampled',
    'imagerotate', 'imagescale', 'imagecrop', 'imageflip', 'imagefilter', 'imageconvolution'];

// CSPRNG calls are always routed to the seeded Mersenne Twister (plain and intercepted runs
// alike): same distributions, but every render is replayable from its seed and per-class
// switches stay exact. Recorded in case metadata as csprng_seeded.
const RNG_WRAPPED = ['random_int', 'random_bytes', 'openssl_random_pseudo_bytes'];

function random_int($min, $max)
{
    if ($max - $min <= \mt_getrandmax()) return \mt_rand($min, $max);
    $span = $max - $min;  // wide ranges: compose two draws
    return $min + (int) ((\mt_rand() * (\mt_getrandmax() + 1) + \mt_rand()) % ($span + 1));
}
function random_bytes($n) { $s = ''; for ($i = 0; $i < $n; $i++) $s .= \chr(\mt_rand(0, 255)); return $s; }
function openssl_random_pseudo_bytes($n, &$strong = null) { $strong = true; return random_bytes($n); }

function rewrite(string $code): string
{
    // Token-based: only real calls are rewritten (never strings, comments, method/static calls,
    // function definitions or `use function` imports).
    $names = \array_flip(State::$gdIntercept ? \array_merge(WRAPPED, RNG_WRAPPED) : RNG_WRAPPED);
    $tokens = \token_get_all($code);
    $skip = [\T_WHITESPACE, \T_COMMENT, \T_DOC_COMMENT];
    $out = '';
    $n = \count($tokens);
    for ($i = 0; $i < $n; $i++) {
        $t = $tokens[$i];
        if (!\is_array($t)) { $out .= $t; continue; }
        [$id, $text] = $t;
        if ($id === \T_STRING || $id === \T_NAME_FULLY_QUALIFIED) {
            $name = \strtolower(\ltrim($text, '\\'));
            if (isset($names[$name])) {
                $j = $i + 1; while ($j < $n && \is_array($tokens[$j]) && \in_array($tokens[$j][0], $skip, true)) $j++;
                $k = $i - 1; while ($k >= 0 && \is_array($tokens[$k]) && \in_array($tokens[$k][0], $skip, true)) $k--;
                $nextIsCall = $j < $n && $tokens[$j] === '(';
                $prev = $k >= 0 ? $tokens[$k] : null;
                $prevId = \is_array($prev) ? $prev[0] : null;
                $blocked = \in_array($prevId, [\T_OBJECT_OPERATOR, \T_NULLSAFE_OBJECT_OPERATOR, \T_DOUBLE_COLON, \T_FUNCTION, \T_NEW, \T_CONST], true);
                if ($nextIsCall && !$blocked) { $out .= '\\BankGD\\' . $name; continue; }
            }
        }
        $out .= $text;
    }
    return $out;
}

/** file:// stream wrapper: rewrites PHP sources under the given roots when they are included. */
final class SourceRewriter
{
    public static array $roots = [];
    public static array $exclude = [];
    public $context;
    private $fh = null;
    private ?string $data = null;
    private int $pos = 0;
    private $dh = null;

    private static function native(callable $f) {
        \stream_wrapper_restore('file');
        try { return $f(); } finally { \stream_wrapper_unregister('file'); \stream_wrapper_register('file', self::class); }
    }
    public static function install(array $roots, array $exclude): void {
        self::$roots = \array_map(fn($r) => \strtolower(\str_replace('\\', '/', \realpath($r))), $roots);
        self::$exclude = \array_map(fn($r) => \strtolower(\str_replace('\\', '/', \realpath($r))), $exclude);
        \stream_wrapper_unregister('file');
        \stream_wrapper_register('file', self::class);
    }
    private static function wanted(string $path): bool {
        $p = \strtolower(\str_replace('\\', '/', $path));
        if (!\preg_match('/\.(php|inc)$/', $p)) return false;
        foreach (self::$exclude as $e) if ($p === $e) return false;
        foreach (self::$roots as $r) if (\str_starts_with($p, $r . '/')) return true;
        return false;
    }
    public function stream_open($path, $mode, $options, &$opened) {
        $real = self::native(fn() => \realpath($path)) ?: $path;
        if (($options & 128 /* STREAM_OPEN_FOR_INCLUDE */) && self::wanted($real)) {
            $src = self::native(fn() => \file_get_contents($path));
            if ($src === false) return false;
            $this->data = rewrite($src);
            return true;
        }
        $this->fh = self::native(fn() => $this->context ? \fopen($path, $mode, (bool) ($options & \STREAM_USE_PATH), $this->context) : \fopen($path, $mode, (bool) ($options & \STREAM_USE_PATH)));
        return $this->fh !== false;
    }
    public function stream_read($n) {
        if ($this->data !== null) { $s = \substr($this->data, $this->pos, $n); $this->pos += \strlen($s); return $s; }
        return \fread($this->fh, $n);
    }
    public function stream_write($d) { return $this->fh ? \fwrite($this->fh, $d) : 0; }
    public function stream_eof() { return $this->data !== null ? $this->pos >= \strlen($this->data) : \feof($this->fh); }
    public function stream_tell() { return $this->data !== null ? $this->pos : \ftell($this->fh); }
    public function stream_seek($o, $w = \SEEK_SET) {
        if ($this->data === null) return \fseek($this->fh, $o, $w) === 0;
        $len = \strlen($this->data);
        $p = $w === \SEEK_SET ? $o : ($w === \SEEK_CUR ? $this->pos + $o : $len + $o);
        if ($p < 0) return false;
        $this->pos = $p; return true;
    }
    public function stream_close() { if ($this->fh) \fclose($this->fh); }
    public function stream_flush() { return $this->fh ? \fflush($this->fh) : true; }
    public function stream_stat() {
        if ($this->data !== null) return ['size' => \strlen($this->data), 'mode' => 0100644];
        return \fstat($this->fh);
    }
    public function stream_lock($op) { return $this->fh ? \flock($this->fh, $op) : true; }
    public function stream_truncate($n) { return $this->fh ? \ftruncate($this->fh, $n) : false; }
    public function stream_set_option($o, $a, $b) { return false; }
    public function stream_metadata($path, $option, $value) {
        return self::native(function () use ($path, $option, $value) {
            switch ($option) {
                case \STREAM_META_TOUCH: return \touch($path, ...((array) $value));
                case \STREAM_META_OWNER_NAME: case \STREAM_META_OWNER: return \chown($path, $value);
                case \STREAM_META_GROUP_NAME: case \STREAM_META_GROUP: return \chgrp($path, $value);
                case \STREAM_META_ACCESS: return \chmod($path, $value);
            }
            return false;
        });
    }
    public function url_stat($path, $flags) {
        return self::native(function () use ($path, $flags) {
            if ($flags & \STREAM_URL_STAT_QUIET) { if (!\file_exists($path)) return false; }
            return ($flags & \STREAM_URL_STAT_LINK) ? @\lstat($path) : @\stat($path);
        });
    }
    public function unlink($p) { return self::native(fn() => \unlink($p)); }
    public function rename($a, $b) { return self::native(fn() => \rename($a, $b)); }
    public function mkdir($p, $m, $o) { return self::native(fn() => \mkdir($p, $m, (bool) ($o & \STREAM_MKDIR_RECURSIVE))); }
    public function rmdir($p, $o) { return self::native(fn() => \rmdir($p)); }
    public function dir_opendir($p, $o) { $this->dh = self::native(fn() => \opendir($p)); return $this->dh !== false; }
    public function dir_readdir() { return \readdir($this->dh); }
    public function dir_rewinddir() { \rewinddir($this->dh); return true; }
    public function dir_closedir() { \closedir($this->dh); return true; }
}

// ---------------------------------------------------------------- results

function idColour(int $id): array { return [($id >> 16) & 255, ($id >> 8) & 255, $id & 255]; }

/** Write <stem>.labels.png (opaque ids), <stem>.labels_translucent.png and <stem>.calls.json. */
function save(string $stem, ?int $outW, ?int $outH): array
{
    $im = State::$emitted ?? State::$lastTouched;
    $note = State::$emitted ? 'emitted' : 'last_touched';
    if ($im && $outW && (\imagesx($im) !== $outW || \imagesy($im) !== $outH)) {
        // the emitted canvas is not the final output size (e.g. adapter re-encoded): fall back
        $im = (State::$lastTouched && \imagesx(State::$lastTouched) === $outW && \imagesy(State::$lastTouched) === $outH) ? State::$lastTouched : null;
        $note = $im ? 'last_touched_size_match' : 'no_matching_canvas';
    }
    $meta = ['label_canvas' => $note, 'copy_writes' => State::$copyWrites, 'unmirrored' => State::$unmirrored,
             'palette_canvas' => $im ? !\imageistruecolor($im) : null, 'class_bits' => CLASS_BITS, 'disabled' => \array_keys(State::$disabled), 'suppressed_pixels' => State::$suppressedPixels,
             'calls' => \count(State::$calls), 'chain_seeded' => State::$chainSeeded, 'next_id' => State::$next];
    if ($im) {
        [$o, $t, $m] = lab($im);
        \imagesavealpha($o, false); \imagesavealpha($t, false); \imagesavealpha($m, false);
        \imagepng($o, $stem . '.labels.png');
        \imagepng($t, $stem . '.labels_translucent.png');
        \imagepng($m, $stem . '.labels_classes.png');
    }
    \file_put_contents($stem . '.calls.json', \json_encode(['meta' => $meta, 'calls' => \array_values(State::$calls)],
        \JSON_UNESCAPED_SLASHES | \JSON_INVALID_UTF8_SUBSTITUTE | \JSON_PARTIAL_OUTPUT_ON_ERROR));
    return $meta;
}
