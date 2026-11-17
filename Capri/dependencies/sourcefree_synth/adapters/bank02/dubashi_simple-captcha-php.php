<?php
// Fontless vector-glyph renderer (symbols drawn as polylines on a point grid).
// Upstream _raw() casts the PHP 8 GdImage object to an array and emits nothing, so the
// adapter reads the private rendered image by reflection and encodes it itself.
// Glyph scope markers around each symbol's polygon (labels for fontless glyphs; no-op without interception).
bank_require_patched(bank_src('src/SimpleCaptcha.php'), [
    're:/(foreach\(\s*\$symbols as \$p=>\$symbol\s*\)\s*\{)/' => '$1 \\BankGD\\glyph_begin((string) $symbol);',
    're:/(\$x \+= \$padding \+ \$charCellWidth;)/' => '\\BankGD\\glyph_end(); $1',
]);
$params = ['width' => bank_int(140, 240), 'height' => bank_int(50, 80), 'type' => IMAGETYPE_PNG,
           'thickness' => bank_int(1, 4), 'dashed' => bank_bool(0.3), 'randPos' => bank_bool(),
           'randPointPos' => bank_bool(), 'randSym' => bank_bool(0.3), 'emboss' => bank_bool(0.2),
           'nightmare' => bank_bool(0.15), 'gradient' => bank_pick([0, 0, 1, 2]), 'noise' => bank_pick([0, 0, 10, 25])];
$text = bank_label('0123456789');
$c = new Dubashi\SimpleCaptcha($params);
$c->create($text);
$prop = new ReflectionProperty(Dubashi\SimpleCaptcha::class, '_image');
$image = $prop->getValue($c);
ob_start(); imagepng($image); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => $text, 'params' => $params, 'bytes' => $bytes];
