<?php
// Script-style library: generate.php runs as-is (its config.php fixes size/colours/counts).
// Controlled text replaces the code after the library's own draws; optional font-bank font.
$GLOBALS['BANK_TEXT'] = bank_label('abcdefghijklmnopqrstuvwxyz');
if (bank_param('font_source', bank_pick(['library', 'bank'])) === 'bank') $GLOBALS['BANK_FONT'] = bank_font(false);
$script = bank_src('generate.php');
chdir(dirname($script));  // require_once('config.php') is relative
bank_require_patched($script, [
    '$_SESSION["captcha"] = $code;' => '$code = $GLOBALS[\'BANK_TEXT\']; $_SESSION["captcha"] = $code;',
    '$font = dirname(__FILE__) . $font_path;' => '$font = $GLOBALS[\'BANK_FONT\'] ?? (dirname(__FILE__) . $font_path);',
]);
$GLOBALS['BANK_RESULT'] = ['label' => $_SESSION['captcha'] ?? null, 'params' => ['upstream_config' => true]];
