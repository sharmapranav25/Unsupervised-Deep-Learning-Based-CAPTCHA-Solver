<?php
// Glyph outlines as one SVG path with randomised curve splitting/approximation (no fonts).
// PHP 8 shim: the answer property starts as "" but is appended to with [] (fatal in PHP 8).
// getInstance() is a non-static singleton factory, so the private constructor is invoked by reflection.
$file = bank_src('SVGCaptcha.php');
chdir(dirname($file));  // constructor does include_once "glyphs.php"
bank_require_patched($file, [
    'private $captcha_answer = "";' => 'private $captcha_answer = array();',
    'implode($this->captcha_answer, "")' => 'implode("", $this->captcha_answer)',  // pre-PHP-8 argument order
    '(microtime() & 1)' => '(mt_rand(0, 1))',  // clock-based coin flip -> seeded, for replayability
    'uniqid($prefix = "gf__")' => '"gf__" . (++$GLOBALS["BANK_UID"])',  // clock-based array key -> counter
    '(intval(time()) & 1)' => '(mt_rand(0, 1))',  // clock-based coin flip -> seeded
    '$selected_keys = array_secure_rand($this->alphabet, $this->numchars, False);' => '$selected_keys = str_split($GLOBALS["BANK_TEXT"]);',
]);
$GLOBALS['BANK_TEXT'] = bank_label('yWGeaHkifbnSXEQ', 2, 8, true);  // glyphs.php keys; upstream uses associative glyphs
$params = ['numchars' => strlen($GLOBALS['BANK_TEXT']), 'width' => bank_param('width', bank_int(240, 320)),
           'height' => bank_param('height', bank_int(100, 140)), 'difficulty' => bank_param('difficulty', bank_pick([1, 2, 3]))];
$rc = new ReflectionClass('SVGCaptcha');
$obj = $rc->newInstanceWithoutConstructor();
$ctor = $rc->getConstructor();
$ctor->invoke($obj, $params['numchars'], $params['width'], $params['height'], $params['difficulty']);
[$answer, $svg] = $obj->getSVGCaptcha();
$GLOBALS['BANK_RESULT'] = ['label' => $answer, 'params' => $params, 'bytes' => $svg,
    'randomness' => 'controlled glyph choice; seeded upstream distortion'];
