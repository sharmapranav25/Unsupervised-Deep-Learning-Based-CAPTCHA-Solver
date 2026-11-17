<?php
// mobicms/captcha (feature bank 01). Fonts come from font bank 02 (a per-case folder of sampled
// fonts) instead of the bundled decorative fonts, whose rights were unresolved in the pilot.
spl_autoload_register(function ($c) {
    $p = 'Mobicms\\Captcha\\';
    if ($c === 'Mobicms\\Captcha\\Image') {  // per-slot font override (params.slot_fonts), after upstream's own choice
        bank_require_patched(bank_src('src/Image.php'), [
            'imagettftext($image, $fontSize, $angle, $xPos, $yPos, $textColor, $font, $letter);'
                => 'imagettftext($image, $fontSize, $angle, $xPos, $yPos, $textColor, \bank_slot_font($key, $font), $letter);',
        ]);
        return;
    }
    if (str_starts_with($c, $p)) require bank_src('src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
});
$text = bank_label('23456789ABCDEGHJKMNPQRSTUVXYZabcdeghjkmnpqrstuvxyz');
$dir = $GLOBALS['BANK_CFG']['tmp'] . '/fonts';
@mkdir($dir);
$n = bank_param('font_count', bank_int(1, 4));
$fonts = [];
for ($i = 0; $i < $n; $i++) {
    $f = bank_font(false);
    $fonts[] = basename(dirname($f));
    copy($f, $dir . '/' . basename(dirname($f)) . '.ttf');
}
$b = new Mobicms\Captcha\Image($text);
$b->fontFolders = [$dir];
$b->fontsTune = [];
$b->fontMix = bank_param('font_mix', $n > 1);
$b->imageWidth = bank_param('width', bank_int(160, 280));
$b->imageHeight = bank_param('height', bank_int(60, 100));
$b->defaultFontSize = bank_param('font_size', bank_int(22, 34));
$bytes = $b->build();
$GLOBALS['BANK_RESULT'] = ['label' => $b->getCode(), 'bytes' => $bytes,
    'params' => ['fonts' => $fonts, 'font_mix' => $b->fontMix, 'w' => $b->imageWidth, 'h' => $b->imageHeight, 'font_size' => $b->defaultFontSize]];
