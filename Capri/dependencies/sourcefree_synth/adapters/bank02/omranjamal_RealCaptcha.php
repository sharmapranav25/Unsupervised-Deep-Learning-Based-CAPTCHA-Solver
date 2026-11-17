<?php
// Package code is MIT; the bundled fonts carry their own (unverified) terms per the upstream header.
// Text is upper-case + digits because several bundled fonts are capitals-only (labels stay exact).
bank_require_patched(bank_src('RealCaptcha.php'), [
    '$settings && $settings["random_length"] ? $settings["random_length"] : $this->settings["random_length"] ? $this->settings["random_length"] : rand(5,9);'
    => '($settings && $settings["random_length"]) ? $settings["random_length"] : ($this->settings["random_length"] ? $this->settings["random_length"] : rand(5,9));',
    // per-slot font override (params.slot_fonts); upstream draws one font for the whole word
    '$font, $raw[$i] );' => '\bank_slot_font($i, $font), $raw[$i] );',
]);
use omranjamal\RealCaptcha\RealCaptcha;
$root = dirname(bank_src('RealCaptcha.php'));
$text = bank_label('ABCDEFGHJKLMNPQRSTUVWXYZ23456789');
$params = ['source' => RealCaptcha::INPUT,
           'width' => bank_param('width', bank_int(300, 500)), 'height' => bank_param('height', bank_int(110, 200)),
           'background_color' => bank_param('background', bank_pick([RealCaptcha::GREY_VARIABLE, [255, 255, 255], [235, 235, 220]])),
           'text_color' => bank_param('text_color', bank_pick([[0, 0, 0], [40, 40, 90], [90, 20, 20]]))];
$fontsDir = $root . '/resources/fonts/';
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') {
    $fontsDir = $GLOBALS['BANK_CFG']['tmp'] . '/fonts/';
    @mkdir($fontsDir);
    for ($i = 0, $n = bank_param('font_count', bank_int(1, 3)); $i < $n; $i++) { $f = bank_font(true); copy($f, $fontsDir . basename(dirname($f)) . '.ttf'); }
}
$c = new RealCaptcha(['fonts_dir' => $fontsDir, 'dictionary_file' => $root . '/resources/dictionary.php']);
$out = $c->generate($text, $params);
ob_start(); imagepng($out->returnGD()); $bytes = ob_get_clean();
$GLOBALS['BANK_RESULT'] = ['label' => $out->text, 'params' => $params, 'bytes' => $bytes];
