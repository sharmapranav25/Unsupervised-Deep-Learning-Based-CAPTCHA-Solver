<?php
// Filament plugin; CaptchaGenerator itself is framework-free and uses its own seeded RNG
// (DeterministicRandom keyed by a seed string), so outputs are replayable from the case seed.
spl_autoload_register(function ($c) {
    $p = 'MortezaAshrafi\\FilamentShieldCaptcha\\';
    if (str_starts_with($c, $p)) require bank_src('src/' . str_replace('\\', '/', substr($c, strlen($p))) . '.php');
});
use MortezaAshrafi\FilamentShieldCaptcha\CaptchaGenerator;
use MortezaAshrafi\FilamentShieldCaptcha\Enums\{CharacterMode, ImageFormat, Theme};
use MortezaAshrafi\FilamentShieldCaptcha\Options\{CaptchaOptions, NoiseOptions};
use MortezaAshrafi\FilamentShieldCaptcha\Support\Color;

$alphabets = ['numeric' => '0123456789', 'alphabetic' => 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz',
              'alphanumeric' => 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789'];
$mode = bank_param('char_mode', bank_pick(array_keys($alphabets)));
$answer = bank_label($alphabets['alphanumeric'] . '01');
$len = max(3, strlen($answer));  // options validator requires length >= 3; drawing uses the answer as given
$fontSource = bank_param('font_source', bank_pick(['library', 'bank']));
$fontPath = $fontSource === 'bank' ? bank_font(false)
    : realpath(bank_root() . '/resources/fonts/' . bank_param('library_font', bank_pick(['NotoSans.ttf', 'Vazirmatn-Regular.ttf'])));
$noise = new NoiseOptions(bank_param('noise_level', bank_int(0, 5)), bank_param('noise_lines', bank_int(0, 8)), bank_param('noise_dots', bank_int(0, 120)));
$opts = new CaptchaOptions(
    bank_param('width', bank_int(160, 260)), bank_param('height', bank_int(50, 90)), min(12, $len), bank_param('font_size', bank_int(20, 30)),
    CharacterMode::from($mode), null, true, $noise,
    new Color(bank_int(230, 255), bank_int(230, 255), bank_int(230, 255)), new Color(bank_int(0, 80), bank_int(0, 80), bank_int(0, 120)),
    new Color(bank_int(10, 40), bank_int(10, 40), bank_int(10, 50)), new Color(bank_int(200, 255), bank_int(200, 255), bank_int(200, 255)),
    $fontPath, ImageFormat::Png, 90, 300, 5,
);
$theme = bank_param('theme', bank_bool(0.25) ? 'dark' : 'light') === 'dark' ? Theme::Dark : Theme::Light;
$uri = (new CaptchaGenerator($opts))->toDataUri($answer, 'bank-' . $GLOBALS['BANK_CFG']['seed'], $theme);
$GLOBALS['BANK_RESULT'] = ['label' => $answer, 'bytes' => $uri, 'randomness' => 'upstream DeterministicRandom keyed by case seed',
    'params' => ['w' => $opts->width, 'h' => $opts->height, 'font_size' => $opts->fontSize, 'font' => basename(dirname($fontPath)) . '/' . basename($fontPath),
                 'noise' => [$noise->level, $noise->lines, $noise->dots], 'theme' => $theme->value]];
