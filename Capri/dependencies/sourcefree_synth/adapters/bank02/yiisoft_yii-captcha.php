<?php
// Yii captcha GD driver (the classic Yii CaptchaAction image: overlapping, offset, rotated glyphs
// in the bundled SpicyRice font). Stand-ins for the three yii base types it references.
namespace yii\base { class Component {} class Application { public function getAlias($a) { return str_replace('@Yiisoft/Yii/Captcha', \bank_root() . '/src', $a); } } }
namespace yii\exceptions { class InvalidConfigException extends \Exception {} }
namespace {
    foreach (['DriverInterface', 'VerifyCodeGeneratorTrait', 'Driver'] as $f) require bank_src("src/$f.php");
    // PHP 8 strict_types shim: round() returns float but imagettftext() needs int.
    bank_require_patched(bank_src('src/GdDriver.php'), ['$y = round($this->height * 27 / 40);' => '$y = (int) round($this->height * 27 / 40);']);
    $d = new Yiisoft\Yii\Captcha\GdDriver(new yii\base\Application());
    if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $d->setFontFile(bank_font(false));
    $d->width = bank_param('width', bank_int(120, 240));
    $d->height = bank_param('height', bank_int(50, 90));
    $d->padding = bank_param('padding', bank_int(0, 6));
    $d->offset = bank_param('offset', bank_int(-4, 2));
    $d->minLength = bank_int(4, 6);
    $d->maxLength = $d->minLength + bank_int(0, 2);
    $d->foreColor = bank_param('fore', bank_pick([0x2040A0, 0x000000, 0x803020, 0x206040, 0x502080]));
    $d->backColor = bank_param('back', bank_pick([0xFFFFFF, 0xF0F0E8, 0xE8F0FF]));
    $d->generateVerifyCode();  // upstream draw consumed; controlled text below
    $code = bank_label('bcdfghjklmnpqrstvwxyzaeiou');
    $bytes = $d->renderImage($code);
    $GLOBALS['BANK_RESULT'] = ['label' => $code, 'bytes' => $bytes,
        'randomness' => 'glyph size/angle use random_int (not replayable)',
        'params' => ['w' => $d->width, 'h' => $d->height, 'padding' => $d->padding, 'offset' => $d->offset,
                     'len' => [$d->minLength, $d->maxLength], 'fore' => dechex($d->foreColor), 'back' => dechex($d->backColor)]];
}
