<?php
namespace SJBR\SrFreecap\ViewHelpers;

/*
 *  Copyright notice
 *
 *  (c) 2013-2026 Stanislas Rolland <typo3AAAA(arobas)sjbr.ca>
 *  All rights reserved
 *
 *  This script is part of the TYPO3 project. The TYPO3 project is
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  The GNU General Public License can be found at
 *  http://www.gnu.org/copyleft/gpl.html.
 *  A copy is found in the textfile GPL.txt and important notices to the license
 *  from the author is found in LICENSE.txt distributed with these scripts.
 *
 *
 *  This script is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  This copyright notice MUST APPEAR in all copies of the script!
 */

use Psr\Http\Message\ServerRequestInterface;
use SJBR\SrFreecap\ViewHelpers\TranslateViewHelper;
use TYPO3\CMS\Core\Context\Context;
use TYPO3\CMS\Core\Session\Backend\Exception\SessionNotFoundException;
use TYPO3\CMS\Core\Utility\ExtensionManagementUtility;
use TYPO3\CMS\Core\Utility\GeneralUtility;
use TYPO3\CMS\Core\Utility\PathUtility;
use TYPO3Fluid\Fluid\Core\ViewHelper\AbstractTagBasedViewHelper;

class ImageViewHelper extends AbstractTagBasedViewHelper
{
	/**
	 * @var string Name of the extension this view helper belongs to
	 */
	protected $extensionName = 'SrFreecap';

	/**
	 * @var string Name of the extension this view helper belongs to
	 */
	protected $extensionKey = 'sr_freecap';

	/**
	 * @var string Name of the plugin this view helper belongs to
	 */
	protected $pluginName = 'tx_srfreecap';

	/**
	 * @var ?ServerRequestInterface
	 */
	protected ?ServerRequestInterface $request = null;

	/**
	 * @var Context
	 */
	protected $context;

	/**
	 * @param Context $context
	 */
	public function injectContext(Context $context): void
	{
		$this->context = $context;
	}

	public function initializeArguments(): void
	{
		parent::initializeArguments();
		$this->registerArgument('suffix', 'string', 'Suffix to be appended to the extenstion key when forming css class names', false, '');
	}

	/**
	 * Render the captcha image html
	 *
	 * @param string $suffix
	 * @return string The html used to render the captcha image
	 */
    public function render($suffix = ''): string
	{
		$request = $this->getRequest();
		// This viewhelper needs a frontend user session
		if (!($request->getAttribute('frontend.user') ?? false)) {
			throw new SessionNotFoundException('No frontend user found in session!', 5427851694);
		}
		$value = '';

		// Get the translation view helper
		$translator = GeneralUtility::makeInstance(TranslateViewHelper::class);

		// Generate the image url
		$pageUid = $request->getAttribute('frontend.page.information')->getId();
		$site = $request->getAttribute('site');
		$fakeId = substr(md5(uniqid(rand())), 0, 5);
		$urlParams = [
			'srFreecap' => '1',
			'pluginName' => 'ImageGenerator',
			'actionName' => 'show',
			'formatName' => 'png',
			'_language' => $request->getAttribute('language')
		];
		$MP = $request->getAttribute('frontend.page.information')->getMountPoint() ?? false;
		if ($MP) {
			$urlParams['MP'] = $MP;
		}
		
		$imageUrl = (string)$site->getRouter($this->context)->generateUri((string)$pageUid, $urlParams) . '&freecapSet=' . $fakeId;

		// Generate the html text
		$value = '<img' . $this->getClassAttribute('image', $suffix) . ' id="tx_srfreecap_captcha_image_' . $fakeId . '"'
			. ' src="' . htmlspecialchars($imageUrl) . '"'
			. ' alt="' . $translator->render('altText') . ' "/>'
			. '<span' . $this->getClassAttribute('cant-read', $suffix) . '>' . $translator->render('cant_read1')
			. ' <a id="tx_srfreecap_captcha_image_' . $fakeId . '_link" data-srfreecap-image="' . $fakeId . '" >'
			. $translator->render('click_here') . '</a>'
			. $translator->render('cant_read2') . '</span>';
		return $value;
	}

	/**
	 * Returns a class attribute with a class-name prefixed with $this->pluginName and with all underscores substituted to dashes (-)
	 *
	 * @param string $class The class name (or the END of it since it will be prefixed by $this->pluginName.'-')
	 * @param string suffix to be appended to the extenstion key when forming css class names
	 * @return string the class attribute with the combined class name (with the correct prefix)
	 */
	protected function getClassAttribute($class, $suffix = '')
	{
		return ' class="' . trim(str_replace('_', '-', $this->pluginName) . ($suffix ? '-' . $suffix . '-' : '-') . $class) . '"';
	}

    private function getRequest(): ServerRequestInterface|null
    {
    	// The rendering context may not be available when outside Extbase context
        if ($this->renderingContext !== null && $this->renderingContext->hasAttribute(ServerRequestInterface::class)) {
            return $this->renderingContext->getAttribute(ServerRequestInterface::class);
        } else if ($this->request !== null) {
        	return $this->request;
        }
        return $GLOBALS['TYPO3_REQUEST'];
    }

	/**
	 * This setter should be called to set the request when this viewhelper is used outside Extbase context
	 *
	 * @param ServerRequestInterface
	 */
    public function setRequest(ServerRequestInterface $request): void
    {
        $this->request = $request;
    }
}