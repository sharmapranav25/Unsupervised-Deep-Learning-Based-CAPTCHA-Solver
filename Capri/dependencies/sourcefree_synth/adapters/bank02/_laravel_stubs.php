<?php
// Minimal stand-ins for the Laravel helpers/facades that CAPTCHA packages touch while rendering.
// Not a framework emulation: only what the bank02 adapters need. Configuration is read from
// $GLOBALS['LARAVEL_CONFIG'] with dot paths; the session is an in-process array.
namespace Illuminate\Support\Facades {
    class Session {
        public static array $d = [];
        public static function put($k, $v = null) { if (is_array($k)) { foreach ($k as $a => $b) self::put($a, $b); return; } \bank_array_set(self::$d, $k, $v); }
        public static function get($k, $def = null) { return \bank_array_get(self::$d, $k, $def); }
        public static function has($k) { return \bank_array_get(self::$d, $k) !== null; }
        public static function forget($k) { unset(self::$d[$k]); }
        public static function pull($k, $def = null) { $v = self::get($k, $def); self::forget($k); return $v; }
        public static function __callStatic($n, $a) { return null; }
    }
    class Cache { public static function __callStatic($n, $a) { return $n === 'remember' ? ($a[2])() : null; } }
}
namespace Illuminate\Support {
    class HtmlString { public function __construct(public string $html = '') {} public function __toString(): string { return $this->html; } }
}
namespace {
    function bank_array_get(array $a, $k, $def = null) {
        foreach (explode('.', (string) $k) as $p) { if (!is_array($a) || !array_key_exists($p, $a)) return $def; $a = $a[$p]; }
        return $a;
    }
    function bank_array_set(array &$a, $k, $v) {
        $ref = &$a;
        foreach (explode('.', (string) $k) as $p) { if (!isset($ref[$p]) || !is_array($ref[$p])) $ref[$p] = []; $ref = &$ref[$p]; }
        $ref = $v;
    }
    if (!function_exists('config')) {
        function config($k = null, $def = null) { return $k === null ? $GLOBALS['LARAVEL_CONFIG'] : bank_array_get($GLOBALS['LARAVEL_CONFIG'] ?? [], $k, $def); }
    }
    if (!function_exists('env')) { function env($k, $def = null) { return $def; } }
    if (!function_exists('route')) { function route($n, $p = []) { return '/' . $n; } }
    if (!function_exists('session')) { function session($k = null, $def = null) { return $k === null ? null : \Illuminate\Support\Facades\Session::get($k, $def); } }
    if (!function_exists('response')) {
        function response($content = '', $status = 200, $headers = []) {
            $GLOBALS['BANK_BYTES'] = $content;
            return new class { public function header(...$a) { return $this; } public function withHeaders(...$a) { return $this; } };
        }
    }
}
