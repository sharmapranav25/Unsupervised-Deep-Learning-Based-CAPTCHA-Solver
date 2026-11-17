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
use TYPO3\CMS\Core\Context\Context;
use TYPO3\CMS\Core\Resource\Exception\InvalidFileException;
use TYPO3\CMS\Core\Session\Backend\Exception\SessionNotFoundException;
use TYPO3\CMS\Core\SystemResource\SystemResourceFactory;
use TYPO3\CMS\Core\SystemResource\Publishing\SystemResourcePublisherInterface;
use TYPO3\CMS\Core\SystemResource\Publishing\UriGenerationOptions;
use TYPO3\CMS\Core\Utility\GeneralUtility;
use TYPO3\CMS\Core\Utility\PathUtility;
use TYPO3\CMS\Extbase\Configuration\ConfigurationManagerInterface;
use TYPO3Fluid\Fluid\Core\ViewHelper\AbstractTagBasedViewHelper;

class AudioViewHelper extends AbstractTagBasedViewHelper
{
	/**
	 * @var string Name of the extension this view helper belongs to
	 */
	protected string $extensionName = 'SrFreecap';

	/**
	 * @var string Name of the extension this view helper belongs to
	 */
	protected string $pluginName = 'tx_srfreecap';

	/**
	 * @var ?ServerRequestInterface
	 */
	protected ?ServerRequestInterface $request = null;

    public function __construct(
    	private readonly ConfigurationManagerInterface $configurationManager,
    	private readonly Context $context,
        private readonly SystemResourceFactory $systemResourceFactory,
        private readonly SystemResourcePublisherInterface $resourcePublisher
    )
    {
    	parent::__construct();
    }

	public function initializeArguments(): void
    {
		parent::initializeArguments();
		$this->registerArgument('suffix', 'string', 'Suffix to be appended to the extenstion key when forming css class names', false, '');
	}

    /**
     * Render the captcha audio rendering request icon
     *
     * @param string $suffix suffix to be appended to the extenstion key when forming css class names
     * @return string The html used to render the captcha audio rendering request icon
     * @throws InvalidFileException
     * @throws SessionNotFoundException
     */
	public function render(string $suffix = ''): string
    {
		$request = $this->getRequest();
		// This viewhelper needs a frontend user session
		if (!($request->getAttribute('frontend.user') ?? false)) {
			throw new SessionNotFoundException('No frontend user found in session!', 1706543285);
		}
		$value = '';
		// Get the plugin configuration
		$settings = $this->configurationManager->getConfiguration(ConfigurationManagerInterface::CONFIGURATION_TYPE_SETTINGS, $this->extensionName, $this->pluginName);
		// Get the translation view helper
		$translator = GeneralUtility::makeInstance(TranslateViewHelper::class);
		// Generate the icon
		if (($settings['accessibleOutput'] ?? false) && ($GLOBALS['TYPO3_CONF_VARS']['SYS']['UTF8filesystem'] ?? false)) {
			$pageUid = $request->getAttribute('frontend.page.information')->getId();
		    $site = $request->getAttribute('site');
			$fakeId = substr(md5(uniqid(rand())), 0, 5);
			$siteURL = $site->getBase();
			$urlParams = [
				'srFreecap' => '1',
				'pluginName' => 'AudioPlayer',
				'actionName' => 'play',
				'formatName' => 'wav',
			    '_language' => $request->getAttribute('language')
			];
			$MP = $request->getAttribute('frontend.page.information')->getMountPoint() ?? false;
			if ($MP) {
				$urlParams['MP'] = $MP;
			}
			$audioURL = (string)$site->getRouter($this->context)->generateUri((string)$pageUid, $urlParams);
			if (isset($settings['accessibleOutputImage']) && $settings['accessibleOutputImage']) {
				$imageResource = $this->systemResourceFactory->createPublicResource($settings['accessibleOutputImage']);
				$imageUri = (string)$this->resourcePublisher->generateUri($imageResource, $request, new UriGenerationOptions(absoluteUri: true));
				$value = '<input id="tx_srfreecap_captcha_audio_' . $fakeId . '_link" type="image" alt="' . $translator->render('click_here_accessible') . '"'
					. ' title="' . $translator->render('click_here_accessible') . '"'
					. ' src="' . $imageUri . '"'
					. ' data-srfreecap-audio="' . $fakeId . '"'
					. ' data-srfreecap-audio-url="' . $audioURL . '"'
					. ' data-srfreecap-audio-noplay="' . $translator->render('noPlayMessage') . '"'
					. ' style="cursor: pointer;"'
					. $this->getClassAttribute('image-accessible', $suffix) . ' />';
			} else {
				$value = '<span id="tx_srfreecap_captcha_playLink_' . $fakeId . '"'
					. $this->getClassAttribute('accessible-link', $suffix) . '>' . $translator->render('click_here_accessible_before_link')
					. '<a id="tx_srfreecap_captcha_audio_' . $fakeId . '_link"'
					. ' data-srfreecap-audio="' . $fakeId . '"'
					. ' data-srfreecap-audio-url="' . $audioURL . '"'
					. ' data-srfreecap-audio-noplay="' .$translator->render('noPlayMessage') . '"'
					. ' href="#" style="cursor: pointer;" title="' . $translator->render('click_here_accessible') . '">'
					. $translator->render('click_here_accessible_link') . '</a>'
					. $translator->render('click_here_accessible_after_link') . '</span>';
			}
			$value .= '<span' . $this->getClassAttribute('accessible', $suffix) . ' id="tx_srfreecap_captcha_playAudio_' . $fakeId . '"></span>';
		}
		return $value;
	}

	/**
	 * Returns a class attribute with a class-name prefixed with $this->pluginName and with all underscores substituted to dashes (-)
	 *
	 * @param string $class The class name (or the END of it since it will be prefixed by $this->pluginName.'-')
	 * @param string $suffix suffix to be appended to the extenstion key when forming css class names
	 * @return string the class attribute with the combined class name (with the correct prefix)
	 */
	protected function getClassAttribute(string $class, string $suffix = ''): string
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
	 */
    public function setRequest(ServerRequestInterface $request): void
    {
        $this->request = $request;
    }
}
