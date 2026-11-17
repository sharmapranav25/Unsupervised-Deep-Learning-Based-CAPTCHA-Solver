<?php
// Symfony bundle's image renderer: GD bitmap font, a math question, per-char vertical jitter.
// Its noise/jitter uses random_int (not replayable). The label is the drawn question text.
require bank_src('src/Service/CaptchaImageRenderer.php');
$a = bank_int(1, 20); $b = bank_int(1, 20); $op = bank_pick(['+', '-', 'x']);
$q = "$a $op $b";
$bytes = (new MulerTech\CaptchaBundle\Service\CaptchaImageRenderer())->render($q);
$GLOBALS['BANK_RESULT'] = ['label' => str_replace(' ', '', $q) . '=?', 'shown_text' => "$q = ?", 'text_kind' => 'math',
    'bytes' => $bytes, 'params' => ['question' => $q], 'randomness' => 'random_int (not replayable)'];
