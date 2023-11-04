/*
Copyright 2011-2024 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstring>
#include <stdexcept>
#include "TextCodec.hpp"
#include "../Global.hpp"
#include "../Magic.hpp"
#include "../util.hpp"

using namespace kanzi;
using namespace std;

// 1024 of the most common English words with at least 2 chars.
char TextCodec::DICT_EN_1024[] =
"TheBeAndOfInToWithItThatForYouHeHaveOnSaidSayAtButWeByHadTheyAsW\
ouldWhoOrCanMayDoThisWasIsMuchAnyFromNotSheWhatTheirWhichGetGive\
HasAreHimHerComeMyOurWereWillSomeBecauseThereThroughTellWhenWork\
ThemYetUpOwnOutIntoJustCouldOverOldThinkDayWayThanLikeOtherHowTh\
enItsPeopleTwoMoreTheseBeenNowWantFirstNewUseSeeTimeManManyThing\
MakeHereWellOnlyHisVeryAfterWithoutAnotherNoAllBelieveBeforeOffT\
houghSoAgainstWhileLastTooDownTodaySameBackTakeEachDifferentWher\
eBetweenThoseEvenSeenUnderAboutOneAlsoFactMustActuallyPreventExp\
ectContainConcernIfSchoolYearGoingCannotDueEverTowardGirlFirmGla\
ssGasKeepWorldStillWentShouldSpendStageDoctorMightJobGoContinueE\
veryoneNeverAnswerFewMeanDifferenceTendNeedLeaveTryNiceHoldSomet\
hingAskWarmLipCoverIssueHappenTurnLookSureDiscoverFightMadDirect\
ionAgreeSomeoneFailRespectNoticeChoiceBeginThreeSystemLevelFeelM\
eetCompanyBoxShowPlayLiveLetterEggNumberOpenProblemFatHandMeasur\
eQuestionCallRememberCertainPutNextChairStartRunRaiseGoalReallyH\
omeTeaCandidateMoneyBusinessYoungGoodCourtFindKnowKindHelpNightC\
hildLotYourUsEyeYesWordBitVanMonthHalfLowMillionHighOrganization\
RedGreenBlueWhiteBlackYourselfEightBothLittleHouseLetDespiteProv\
ideServiceHimselfFriendDescribeFatherDevelopmentAwayKillTripHour\
GameOftenPlantPlaceEndAmongSinceStandDesignParticularSuddenlyMem\
berPayLawBookSilenceAlmostIncludeAgainEitherToolFourOnceLeastExp\
lainIdentifyUntilSiteMinuteCoupleWeekMatterBringDetailInformatio\
nNothingAnythingEverythingAgoLeadSometimesUnderstandWhetherNatur\
eTogetherFollowParentStopIndeedDifficultPublicAlreadySpeakMainta\
inRemainHearAllowMediaOfficeBenefitDoorHugPersonLaterDuringWarHi\
storyArgueWithinSetArticleStationMorningWalkEventWinChooseBehavi\
orShootFireFoodTitleAroundAirTeacherGapSubjectEnoughProveAcrossA\
lthoughHeadFootSecondBoyMainLieAbleCivilTableLoveProcessOfferStu\
dentConsiderAppearStudyBuyNearlyHumanEvidenceTextMethodIncluding\
SendRealizeSenseBuildControlAudienceSeveralCutCollegeInterestSuc\
cessSpecialRiskExperienceBehindBetterResultTreatFiveRelationship\
AnimalImproveHairStayTopReducePerhapsLateWriterPickElseSignifica\
ntChanceHotelGeneralRockRequireAlongFitThemselvesReportCondition\
ReachTruthEffortDecideRateEducationForceGardenDrugLeaderVoiceQui\
teWholeSeemMindFinallySirReturnFreeStoryRespondPushAccordingBrot\
herLearnSonHopeDevelopFeelingReadCarryDiseaseRoadVariousBallCase\
OperationCloseVisitReceiveBuildingValueResearchFullModelJoinSeas\
onKnownDirectorPositionPlayerSportErrorRecordRowDataPaperTheoryS\
paceEveryFormSupportActionOfficialWhoseIdeaHappyHeartBestTeamPro\
jectHitBaseRepresentTownPullBusMapDryMomCatDadRoomSmileFieldImpa\
ctFundLargeDogHugePrepareEnvironmentalProduceHerselfTeachOilSuch\
SituationTieCostIndustrySkinStreetImageItselfPhonePriceWearMostS\
unSoonClearPracticePieceWaitRecentImportantProductLeftWallSeries\
NewsShareMovieKidNorSimplyWifeOntoCatchMyselfFineComputerSongAtt\
entionDrawFilmRepublicanSecurityScoreTestStockPositiveCauseCentu\
ryWindowMemoryExistListenStraightCultureBillionFormerDecisionEne\
rgyMoveSummerWonderRelateAvailableLineLikelyOutsideShotShortCoun\
tryRoleAreaSingleRuleDaughterMarketIndicatePresentLandCampaignMa\
terialPopulationEconomyMedicalHospitalChurchGroundThousandAuthor\
ityInsteadRecentlyFutureWrongInvolveLifeHeightIncreaseRightBankC\
ulturalCertainlyWestExecutiveBoardSeekLongOfficerStatementRestBa\
yDealWorkerResourceThrowForwardPolicyScienceEyesBedItemWeaponFil\
lPlanMilitaryGunHotHeatAddressColdFocusForeignTreatmentBloodUpon\
CourseThirdWatchAffectEarlyStoreThusSoundEverywhereBabyAdministr\
ationMouthPageEnterProbablyPointSeatNaturalRaceFarChallengePassA\
pplyMailUsuallyMixToughClearlyGrowFactorStateLocalGuyEastSaveSou\
thSceneMotherCareerQuicklyCentralFaceIceAboveBeyondPictureNetwor\
kManagementIndividualWomanSizeSpeedBusySeriousOccurAddReadySignC\
ollectionListApproachChargeQualityPressureVoteNotePartRealWebCur\
rentDetermineTrueSadWhateverBreakWorryCupParticularlyAmountAbili\
tyEatRecognizeSitCharacterSomebodyLossDegreeEffectAttackStaffMid\
dleTelevisionWhyLegalCapitalTradeElectionEverybodyDropMajorViewS\
tandardBillEmployeeDiscussionOpportunityAnalysisTenSuggestLawyer\
HusbandSectionBecomeSkillSisterStyleCrimeProgramCompareCapMissBa\
dSortTrainingEasyNearRegionStrategyPurposePerformTechnologyEcono\
micBudgetExampleCheckEnvironmentDoneDarkTermRatherLaughGuessCarL\
owerHangPastSocialForgetHundredRemoveManagerEnjoyExactlyDieFinal\
MaybeHealthFloorChangeAmericanPoorFunEstablishTrialSpringDinnerB\
igThankProtectAvoidImagineTonightStarArmFinishMusicOwnerCryArtPr\
ivateOthersSimplePopularReflectEspeciallySmallLightMessageStepKe\
yPeaceProgressMadeSideGreatFixInterviewManageNationalFishLoseCam\
eraDiscussEqualWeightPerformanceSevenWaterProductionPersonalCell\
PowerEveningColorInsideBarUnitLessAdultWideRangeMentionDeepEdgeS\
trongHardTroubleNecessarySafeCommonFearFamilySeaDreamConferenceR\
eplyPropertyMeetingAlwaysStuffAgencyDeathGrowthSellSoldierActHea\
vyWetBagMarriageDeadSingRiseDecadeWhomFigurePoliceBodyMachineCat\
egoryAheadFrontCareOrderRealityPartnerYardBeatViolenceTotalDefen\
seWriteConsumerCenterGroupThoughtModernTaskCoachReasonAgeFingerS\
pecificConnectionWishResponsePrettyMovementCardLogNumberSumTreeE\
ntireCitizenThroughoutPetSimilarVictimNewspaperThreatClassShakeS\
ourceAccountPainFallRichPossibleAcceptSolidTravelTalkSaidCreateN\
onePlentyPeriodDefineNormalRevealDrinkAuthorServeNameMomentAgent\
DocumentActivityAnywayAfraidTypeActiveTrainInterestingRadioDange\
rGenerationLeafCopyMatchClaimAnyoneSoftwarePartyDeviceCodeLangua\
geLinkHoweverConfirmCommentCityAnywhereSomewhereDebateDriveHighe\
rBeautifulOnlineFanPriorityTraditionalSixUnited";


DictEntry TextCodec::STATIC_DICTIONARY[1024] = {};
bool TextCodec::DELIMITER_CHARS[256] = {};
bool TextCodec::TEXT_CHARS[256] = {};
const bool TextCodec::INIT = TextCodec::init(TextCodec::DELIMITER_CHARS, TextCodec::TEXT_CHARS);
const int TextCodec::STATIC_DICT_WORDS = TextCodec::createDictionary(DICT_EN_1024, sizeof(DICT_EN_1024), STATIC_DICTIONARY, 1024, 0);

bool TextCodec::init(bool delims[256], bool text[256])
{
    for (int i = 0; i < 256; i++) {
        if ((i >= ' ') && (i <= '/')) // [ !"#$%&'()*+,-./]
            delims[i] = true;
        else if ((i >= ':') && (i <= '?')) // [:;<=>?]
            delims[i] = true;
        else {
            switch (i) {
            case '\n':
            case '\r':
            case '\t':
            case '_':
            case '|':
            case '{':
            case '}':
            case '[':
            case ']':
                delims[i] = true;
                break;
            default:
                delims[i] = false;
            }
        }

        text[i] = isUpperCase(byte(i)) || isLowerCase(byte(i));
    }

    return true;
}

// Create dictionary from array of words
int TextCodec::createDictionary(char words[], int dictSize, DictEntry dict[], int maxWords, int startWord)
{
    int delimAnchor = 0;
    int h = HASH1;
    int nbWords = startWord;
    byte* src = reinterpret_cast<byte*>(words);

    for (int i = 0; ((i < dictSize) && (nbWords < maxWords)); i++) {
        if (isText(src[i]) == false)
            continue;

        if (isUpperCase(src[i])) {
            if (i > delimAnchor) {
                dict[nbWords] = DictEntry(&src[delimAnchor], h, nbWords, i - delimAnchor);
                nbWords++;
                delimAnchor = i;
                h = HASH1;
            }

            src[i] ^= byte(0x20);
        }

        h = h * HASH1 ^ int(src[i]) * HASH2;
    }

    if (nbWords < maxWords) {
        dict[nbWords] = DictEntry(&src[delimAnchor], h, nbWords, dictSize - 1 - delimAnchor);
        nbWords++;
    }

    return nbWords;
}

// Analyze the block and return an 8-bit status (see MASK flags constants)
// The goal is to detect text data amenable to pre-processing.
byte TextCodec::computeStats(const byte block[], int count, uint freqs0[], bool strict)
{
    if (strict == false) {
        // This is going to fail if the block is not the first of the file.
        // But this is a cheap test, good enough for fast mode.
        if (Magic::getType(block) != Magic::NO_MAGIC)
            return TextCodec::MASK_NOT_TEXT;
    }

    uint* freqs1 = new uint[65536];
    memset(&freqs1[0], 0, 65536 * sizeof(uint));
    uint f0[256] = { 0 };
    uint f1[256] = { 0 };
    uint f3[256] = { 0 };
    uint f2[256] = { 0 };
    uint8 prv = 0;
    const uint8* data = reinterpret_cast<const uint8*>(&block[0]);
    const int count4 = count & -4;

    // Unroll loop
    for (int i = 0; i < count4; i += 4) {
        const uint8 cur0 = data[i];
        const uint8 cur1 = data[i + 1];
        const uint8 cur2 = data[i + 2];
        const uint8 cur3 = data[i + 3];
        f0[cur0]++;
        f1[cur1]++;
        f2[cur2]++;
        f3[cur3]++;
        freqs1[(prv  * 256) + cur0]++;
        freqs1[(cur0 * 256) + cur1]++;
        freqs1[(cur1 * 256) + cur2]++;
        freqs1[(cur2 * 256) + cur3]++;
        prv = cur3;
    }

    for (int i = count4; i < count; i++) {
        freqs0[data[i]]++;
        freqs1[(prv * 256) + data[i]]++;
        prv = data[i];
    }

    for (int i = 0; i < 256; i++) {
        freqs0[i] += (f0[i] + f1[i] + f2[i] + f3[i]);
    }

    const int cr = int(CR);
    const int lf = int(LF);
    int nbTextChars = freqs0[cr] + freqs0[lf];
    int nbASCII = 0;

    for (int i = 0; i < 128; i++) {
        if (isText(byte(i)) == true)
            nbTextChars += freqs0[i];

        nbASCII += freqs0[i];
    }

    // Not text (crude thresholds)
    const int nbBinChars = count - nbASCII;
    bool notText = nbBinChars > (count >> 2);

    if (notText == false) {
        if (strict == true) {
            notText = ((nbTextChars < (count >> 2)) || (freqs0[0] >= uint(count / 100)) || ((nbASCII / 95) < (count / 100)));
        } else {
            notText = ((nbTextChars < (count >> 1)) || (freqs0[32] < uint(count / 50)));
        }
    }

    byte res = byte(0);

    if (notText == true) {
        res |= detectType(freqs0, freqs1, count);
        delete[] freqs1;
        return res;
    }

    if (nbBinChars <= count - count / 10) {
        // Check if likely XML/HTML
        // Another crude test: check that the frequencies of < and > are similar
        // and 'high enough'. Also check it is worth to attempt replacing ampersand sequences.
        // Getting this flag wrong results in a very small compression speed degradation.
        const int f60 = freqs0[60]; // '<'
        const int f62 = freqs0[62]; // '>'
        const int f38 = freqs1[38 * 256 + 97]  + freqs1[38 * 256 + 103] + 
                        freqs1[38 * 256 + 108] + freqs1[38 * 256 + 113]; // '&a', '&g', '&l', '&q'
        const int minFreq = max((count - nbBinChars) >> 9, 2);

        if ((f60 >= minFreq) && (f62 >= minFreq) && (f38 > 0)) {
            if (f60 < f62) {
                if (f60 >= (f62 - f62 / 100))
                    res |= TextCodec::MASK_XML_HTML;
            }
            else if (f62 < f60) {
                if (f62 >= (f60 - f60 / 100))
                    res |= TextCodec::MASK_XML_HTML;
            }
            else {
                res |= TextCodec::MASK_XML_HTML;
            }
        }
    }

    // Check CR+LF matches
    if ((freqs0[cr] != 0) && (freqs0[cr] == freqs0[lf])) {
        res |= TextCodec::MASK_CRLF;

        for (int i = 0; i < 256; i++) {
            if ((i != lf) && (freqs1[(cr * 256) + i]) != 0) {
                res &= ~TextCodec::MASK_CRLF;
                break;
            }

            if ((i != cr) && (freqs1[(i * 256) + lf]) != 0) {
                res &= ~TextCodec::MASK_CRLF;
                break;
            }
        }
    }

    delete[] freqs1;
    return res;
}

byte TextCodec::detectType(uint freqs0[], uint freqs1[], int count) {
    Global::DataType dt = Global::detectSimpleType(count, freqs0);
	
    if (dt != Global::UNDEFINED)
       return TextCodec::MASK_NOT_TEXT | byte(dt);

    // Check UTF-8
    // See Unicode 14 Standard - UTF-8 Table 3.7
    // U+0000..U+007F          00..7F
    // U+0080..U+07FF          C2..DF 80..BF
    // U+0800..U+0FFF          E0 A0..BF 80..BF
    // U+1000..U+CFFF          E1..EC 80..BF 80..BF
    // U+D000..U+D7FF          ED 80..9F 80..BF 80..BF
    // U+E000..U+FFFF          EE..EF 80..BF 80..BF
    // U+10000..U+3FFFF        F0 90..BF 80..BF 80..BF
    // U+40000..U+FFFFF        F1..F3 80..BF 80..BF 80..BF
    // U+100000..U+10FFFF      F4 80..8F 80..BF 80..BF

    if ((freqs0[0xC0] > 0) || (freqs0[0xC1] > 0))
        return TextCodec::MASK_NOT_TEXT;

    for (int i = 0xF5; i <= 0xFF; i++) {
        if (freqs0[i] > 0)
            return TextCodec::MASK_NOT_TEXT;
    }
   
    int sum = 0;

    for (int i = 0; i < 256; i++) {
        // Exclude < 0xE0A0 || > 0xE0BF
        if (((i < 0xA0) || (i > 0xBF)) && (freqs1[(0xE0 << 8) + i] > 0))
            return TextCodec::MASK_NOT_TEXT;

        // Exclude < 0xED80 || > 0xEDE9F
        if (((i < 0x80) || (i > 0x9F)) && (freqs1[(0xED << 8) + i] > 0))
            return TextCodec::MASK_NOT_TEXT;

        // Exclude < 0xF090 || > 0xF0BF
        if (((i < 0x90) || (i > 0xBF)) && (freqs1[(0xF0 << 8) + i] > 0))
            return TextCodec::MASK_NOT_TEXT;

        // Exclude < 0xF480 || > 0xF4BF
        if (((i < 0x80) || (i > 0xBF)) && (freqs1[(0xF4 << 8) + i] > 0))
            return TextCodec::MASK_NOT_TEXT;

        // Count non-primary bytes
        if ((i >= 0x80) && (i <= 0xBF))
           sum += freqs0[i];
    }

    // Another ad-hoc threshold
    return (sum < count / 4) ? TextCodec::MASK_NOT_TEXT : TextCodec::MASK_NOT_TEXT | byte(Global::UTF8);
}


TextCodec::TextCodec()
{
    _delegate = new TextCodec1();
}

TextCodec::TextCodec(Context& ctx)
{
    int encodingType = ctx.getInt("textcodec", 1);
    _delegate = (encodingType == 1) ? static_cast<Transform<byte>*>(new TextCodec1(ctx)) :
        static_cast<Transform<byte>*>(new TextCodec2(ctx));
}

bool TextCodec::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count) THROW
{
    if (count == 0)
        return true;

    if ((count < MIN_BLOCK_SIZE) || (count > MAX_BLOCK_SIZE))
        return false;

    if (!SliceArray<byte>::isValid(input))
       throw invalid_argument("TextCodec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("TextCodec: Invalid output block");

    if (input._array == output._array)
        return false;

    return _delegate->forward(input, output, count);
}

bool TextCodec::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count) THROW
{
    if (count == 0)
        return true;

    if (count > MAX_BLOCK_SIZE) // ! no min
        return false;

    if (!SliceArray<byte>::isValid(input))
        throw invalid_argument("TextCodec: Invalid input block");

    if (!SliceArray<byte>::isValid(output))
        throw invalid_argument("TextCodec: Invalid output block");

    if (input._array == output._array)
        return false;

    return _delegate->inverse(input, output, count);
}

TextCodec1::TextCodec1()
{
    _logHashSize = TextCodec::LOG_HASHES_SIZE;
    _dictSize = 1 << 13;
    _dictMap = nullptr;
    _dictList = nullptr;
    _hashMask = (1 << _logHashSize) - 1;
    _staticDictSize = TextCodec::STATIC_DICT_WORDS;
    _isCRLF = false;
    _escapes[0] = TextCodec::ESCAPE_TOKEN2;
    _escapes[1] = TextCodec::ESCAPE_TOKEN1;
    _pCtx = nullptr;
}

TextCodec1::TextCodec1(Context& ctx)
{
    // Actual block size
    const int blockSize = ctx.getInt("blockSize", 0);
    const int log = (blockSize >= 8) ? max(min(Global::log2(blockSize / 8), 26), 13) : 13;
    _logHashSize = (ctx.getInt("extra", 0) == 0) ? log : log + 1;
    _dictSize = 1 << 13;
    _dictMap = nullptr;
    _dictList = nullptr;
    _hashMask = (1 << _logHashSize) - 1;
    _staticDictSize = TextCodec::STATIC_DICT_WORDS;
    _isCRLF = false;
    _escapes[0] = TextCodec::ESCAPE_TOKEN2;
    _escapes[1] = TextCodec::ESCAPE_TOKEN1;
    _pCtx = &ctx;
}

void TextCodec1::reset(int count)
{
    // Select an appropriate initial dictionary size
    const int log = (count < 1024) ? 13 : max(min(Global::log2(count / 128), 18), 13);
    _dictSize = max(TextCodec::STATIC_DICT_WORDS + 2, 1 << log);
    const int mapSize = 1 << _logHashSize;

    if (_dictMap == nullptr)
        _dictMap = new DictEntry*[mapSize];

    for (int i = 0; i < mapSize; i++)
        _dictMap[i] = nullptr;

    if (_dictList == nullptr) {
        _dictList = new DictEntry[_dictSize];
#if __cplusplus >= 201103L
        memcpy(&_dictList[0], &TextCodec::STATIC_DICTIONARY[0], sizeof(TextCodec::STATIC_DICTIONARY));
#else
	for (int i = 0; i < TextCodec::STATIC_DICT_WORDS; i++)
        _dictList[i] = TextCodec::STATIC_DICTIONARY[i];
#endif

        // Add special entries at end of static dictionary
        _staticDictSize = TextCodec::STATIC_DICT_WORDS;
        _dictList[_staticDictSize] = DictEntry(&_escapes[0], 0, _staticDictSize, 1);
        _dictList[_staticDictSize + 1] = DictEntry(&_escapes[1], 0, _staticDictSize + 1, 1);
        _staticDictSize += 2;
    }

    for (int i = 0; i < _staticDictSize; i++)
        _dictMap[_dictList[i]._hash & _hashMask] = &_dictList[i];

    // Pre-allocate all dictionary entries
    for (int i = _staticDictSize; i < _dictSize; i++)
        _dictList[i] = DictEntry(nullptr, 0, i);
}

bool TextCodec1::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (output._length - output._index < getMaxEncodedLength(count))
        return false;

    byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    int srcIdx = 0;
    int dstIdx = 0;

    if (_pCtx != nullptr) {
        Global::DataType dt = (Global::DataType) _pCtx->getInt("dataType", Global::UNDEFINED);

        // Filter out most types. Still check binaries which may contain significant parts of text
        if ((dt != Global::UNDEFINED) && (dt != Global::TEXT) && (dt != Global::BIN))
            return false;
    }

    uint freqs[256] = { 0 };
    byte mode = TextCodec::computeStats(&src[srcIdx], count, freqs, true);

    // Not text ?
    if ((mode & TextCodec::MASK_NOT_TEXT) != byte(0)) {
        if (_pCtx != nullptr)
            _pCtx->putInt("dataType", Global::DataType(mode & TextCodec::MASK_DT));

        return false;
    }

    if (_pCtx != nullptr)
       _pCtx->putInt("dataType", Global::TEXT);

    reset(count);
    const int srcEnd = count;
    const int dstEnd = getMaxEncodedLength(count);
    const int dstEnd4 = dstEnd - 4;
    int emitAnchor = 0; // never less than 0
    int words = _staticDictSize;

    // DOS encoded end of line (CR+LF) ?
    _isCRLF = int(mode & TextCodec::MASK_CRLF) != 0;
    dst[dstIdx++] = mode;
    bool res = true;

    while ((srcIdx < srcEnd) && (src[srcIdx] == TextCodec::SP)) {
        dst[dstIdx++] = TextCodec::SP;
        srcIdx++;
        emitAnchor++;
    }

    int delimAnchor = TextCodec::isText(src[srcIdx]) ? srcIdx - 1 : srcIdx; // previous delimiter

    while (srcIdx < srcEnd) {
        if (TextCodec::isText(src[srcIdx])) {
            srcIdx++;
            continue;
        }

        if ((srcIdx > delimAnchor + 2) && TextCodec::isDelimiter(src[srcIdx])) { // At least 2 letters
            const byte val = src[delimAnchor + 1];
            const int length = srcIdx - delimAnchor - 1;

            if (length <= TextCodec::MAX_WORD_LENGTH) {
                // Compute hashes
                // h1 -> hash of word chars
                // h2 -> hash of word chars with first char case flipped
                int h1 = TextCodec::HASH1;
                h1 = h1 * TextCodec::HASH1 ^ int(val) * TextCodec::HASH2;
                int h2 = TextCodec::HASH1;
                h2 = h2 * TextCodec::HASH1 ^ (int(val) ^ 0x20) * TextCodec::HASH2;

                for (int i = delimAnchor + 2; i < srcIdx; i++) {
                    h1 = h1 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;
                    h2 = h2 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;
                }

                // Check word in dictionary
                DictEntry* pe = nullptr;
                prefetchRead(&_dictMap[h1 & _hashMask]);
                DictEntry* pe1 = _dictMap[h1 & _hashMask];

                // Check for hash collisions
                if ((pe1 != nullptr) && (pe1->_hash == h1) && ((pe1->_data >> 24) == length))
                    pe = pe1;
                else {
                    prefetchRead(&_dictMap[h2 & _hashMask]);
                    DictEntry* pe2 = _dictMap[h2 & _hashMask];

                    if ((pe2 != nullptr) && (pe2->_hash == h2) && ((pe2->_data >> 24) == length))
                        pe = pe2;
                }

                if (pe != nullptr) {
                    if (!TextCodec::sameWords(&pe->_ptr[1], &src[delimAnchor + 2], length - 1))
                        pe = nullptr;
                }

                if (pe == nullptr) {
                    // Word not found in the dictionary or hash collision.
                    // Replace entry if not in static dictionary
                    if (((length > 3) || ((length == 3) && (words < TextCodec::THRESHOLD2))) && (pe1 == nullptr)) {
                        DictEntry* pe3 = &_dictList[words];

                        if ((pe3->_data & TextCodec::MASK_LENGTH) >= _staticDictSize) {
                            // Reuse old entry
                            _dictMap[pe3->_hash & _hashMask] = nullptr;
                            pe3->_ptr = &src[delimAnchor + 1];
                            pe3->_hash = h1;
                            pe3->_data = (length << 24) | words;
                        }

                        // Update hash map
                        _dictMap[h1 & _hashMask] = pe3;
                        words++;

                        // Dictionary full ? Expand or reset index to end of static dictionary
                        if (words >= _dictSize) {
                            if (expandDictionary() == false)
                                words = _staticDictSize;
                        }
                    }
                }
                else {
                    // Word found in the dictionary
                    // Skip space if only delimiter between 2 word references
                    if ((emitAnchor != delimAnchor) || (src[delimAnchor] != byte(' '))) {
                        const int dIdx = emitSymbols(&src[emitAnchor], &dst[dstIdx], delimAnchor + 1 - emitAnchor, dstEnd - dstIdx);

                        if (dIdx < 0) {
                            res = false;
                            break;
                        }

                        dstIdx += dIdx;
                    }

                    if (dstIdx >= dstEnd4) {
                        res = false;
                        break;
                    }

                    dst[dstIdx++] = (pe == pe1) ? TextCodec::ESCAPE_TOKEN1 : TextCodec::ESCAPE_TOKEN2;
                    dstIdx += emitWordIndex(&dst[dstIdx], pe->_data & TextCodec::MASK_LENGTH);
                    emitAnchor = delimAnchor + 1 + int(pe->_data >> 24);
                }
            }
        }

        // Reset delimiter position
        delimAnchor = srcIdx;
        srcIdx++;
    }

    if (res == true) {
        // Emit last symbols
        const int dIdx = emitSymbols(&src[emitAnchor], &dst[dstIdx], srcEnd - emitAnchor, dstEnd - dstIdx);

        if (dIdx < 0)
            res = false;
        else
            dstIdx += dIdx;

        res &= (srcIdx == srcEnd);
    }

    output._index += dstIdx;
    input._index += srcIdx;
    return res;
}

bool TextCodec1::expandDictionary()
{
    if (_dictSize >= TextCodec::MAX_DICT_SIZE)
        return false;

    DictEntry* newDict = new DictEntry[_dictSize * 2];
    memcpy(static_cast<void*>(&newDict[0]), &_dictList[0], sizeof(DictEntry) * _dictSize);

    for (int i = _dictSize; i < _dictSize * 2; i++)
        newDict[i] = DictEntry(nullptr, 0, i);

    delete[] _dictList;
    _dictList = newDict;

    // Reset map (values must point to addresses of new DictEntry items)
    for (int i = 0; i < _dictSize; i++) {
        _dictMap[_dictList[i]._hash & _hashMask] = &_dictList[i];
    }

    _dictSize <<= 1;
    return true;
}

int TextCodec1::emitSymbols(const byte src[], byte dst[], const int srcEnd, const int dstEnd)
{
    int dstIdx = 0;

    for (int i = 0; i < srcEnd; i++) {
        if (dstIdx >= dstEnd)
            return -1;

// Work around incorrect warning by GCC 7.x.x with C++17
#ifdef __GNUC__
    #if (__GNUC__ == 7) && (__cplusplus > 201402L)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wswitch"
    #endif
#endif

        const byte cur = src[i];

        switch (cur) {
        case TextCodec::ESCAPE_TOKEN1:
        case TextCodec::ESCAPE_TOKEN2: {
            // Emit special word
            dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
            const int idx = (cur == TextCodec::ESCAPE_TOKEN1) ? _staticDictSize - 1 : _staticDictSize - 2;
            int lenIdx = 1;

            if (idx >= TextCodec::THRESHOLD1)
                lenIdx = (idx >= TextCodec::THRESHOLD2) ? 3 : 2;

            if (dstIdx + lenIdx >= dstEnd)
                return -1;

            dstIdx += emitWordIndex(&dst[dstIdx], idx);
            break;
        }

        case TextCodec::CR:
            if (_isCRLF == false)
                dst[dstIdx++] = cur;

            break;

        default:
            dst[dstIdx++] = cur;
        }
    }

// Work around incorrect warning by GCC 7.x.x with C++17
#ifdef __GNUC__
    #if (__GNUC__ == 7) && (__cplusplus > 201402L)
        #pragma GCC diagnostic pop
    #endif
#endif

    return dstIdx;
}

int TextCodec1::emitWordIndex(byte dst[], int val)
{
    // Emit word index (varint 5 bits + 7 bits + 7 bits)
    if (val >= TextCodec::THRESHOLD1) {
        int dstIdx = 0;

        if (val >= TextCodec::THRESHOLD2)
            dst[dstIdx++] = byte(0xE0 | (val >> 14));

        dst[dstIdx] = byte(0x80 | (val >> 7));
        dst[dstIdx + 1] = byte(0x7F & val);
        return dstIdx + 2;
    }

    dst[0] = byte(val);
    return 1;
}

bool TextCodec1::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    reset(output._length);
    byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    _isCRLF = int(src[0] & TextCodec::MASK_CRLF) != 0;
    int srcIdx = 1;
    int dstIdx = 0;
    const int srcEnd = count;
    const int dstEnd = output._length;
    int delimAnchor = TextCodec::isText(src[srcIdx]) ? srcIdx - 1 : srcIdx; // previous delimiter
    int words = _staticDictSize;
    bool wordRun = false;

    while ((srcIdx < srcEnd) && (dstIdx < dstEnd)) {
        const byte cur = src[srcIdx];

        if (TextCodec::isText(cur)) {
            dst[dstIdx] = cur;
            srcIdx++;
            dstIdx++;
            continue;
        }

        if ((srcIdx > delimAnchor + 3) && TextCodec::isDelimiter(cur)) {
            const int length = srcIdx - delimAnchor - 1; // length > 2

            if (length <= TextCodec::MAX_WORD_LENGTH) {
                int h1 = TextCodec::HASH1;

                for (int i = delimAnchor + 1; i < srcIdx; i++)
                    h1 = h1 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;

                // Lookup word in dictionary
                DictEntry* pe = nullptr;
                DictEntry* pe1 = _dictMap[h1 & _hashMask];

                // Check for hash collisions
                if ((pe1 != nullptr) && (pe1->_hash == h1) && ((pe1->_data >> 24) == length)) {
                    if (TextCodec::sameWords(&pe1->_ptr[1], &src[delimAnchor + 2], length - 1))
                        pe = pe1;
                }

                if (pe == nullptr) {
                    // Word not found in the dictionary or hash collision.
                    // Replace entry if not in static dictionary
                    if (((length > 3) || (words < TextCodec::THRESHOLD2)) && (pe1 == nullptr)) {
                        DictEntry& e = _dictList[words];

                        if ((e._data & TextCodec::MASK_LENGTH) >= _staticDictSize) {
                            // Reuse old entry
                            _dictMap[e._hash & _hashMask] = nullptr;
                            e._ptr = &src[delimAnchor + 1];
                            e._hash = h1;
                            e._data = (length << 24) | words;
                        }

                        _dictMap[h1 & _hashMask] = &e;
                        words++;

                        // Dictionary full ? Expand or reset index to end of static dictionary
                        if (words >= _dictSize) {
                            if (expandDictionary() == false)
                                words = _staticDictSize;
                        }
                    }
                }
            }
        }

        srcIdx++;

        if ((cur == TextCodec::ESCAPE_TOKEN1) || (cur == TextCodec::ESCAPE_TOKEN2)) {
            // Word in dictionary
            // Read word index (varint 5 bits + 7 bits + 7 bits)
            int idx = int(src[srcIdx++]);

            if (idx >= 128) {
                const int idx2 = int(src[srcIdx++]);

                if (idx2 >= 128) {
                    idx = ((idx & 0x1F) << 14) | ((idx2 & 0x7F) << 7) | int(src[srcIdx]);
                    srcIdx++;
                }
                else {
                    idx = ((idx & 0x7F) << 7) | idx2;
	        }

                if (idx >= _dictSize)
                    break;
            }

            const int length = _dictList[idx]._data >> 24;

            // Sanity check
            if (dstIdx + length >= dstEnd)
                break;

            // Emit word
            if (length > 1) {
                // Add space if only delimiter between 2 words (not an escaped delimiter)
                if (wordRun == true)
                    dst[dstIdx++] = TextCodec::SP;

                // Regular word entry
                wordRun = true;
                delimAnchor = srcIdx;
            }
            else {
                if (length == 0)
                   break;

                // Escape entry
                wordRun = false;
                delimAnchor = srcIdx - 1;
            }

            memcpy(&dst[dstIdx], _dictList[idx]._ptr, length);

            // Flip case of first character ?
            if (cur == TextCodec::ESCAPE_TOKEN2)
               dst[dstIdx] ^= byte(0x20);

            dstIdx += length;
        }
        else {
            wordRun = false;
            delimAnchor = srcIdx - 1;

            if ((_isCRLF == true) && (cur == TextCodec::LF))
                dst[dstIdx++] = TextCodec::CR;

            dst[dstIdx++] = cur;
        }
    }

    output._index += dstIdx;
    input._index += srcIdx;
    return srcIdx == srcEnd;
}

TextCodec2::TextCodec2()
{
    _logHashSize = TextCodec::LOG_HASHES_SIZE;
    _dictSize = 1 << 13;
    _dictMap = nullptr;
    _dictList = nullptr;
    _hashMask = (1 << _logHashSize) - 1;
    _staticDictSize = TextCodec::STATIC_DICT_WORDS;
    _isCRLF = false;
    _pCtx = nullptr;
}

TextCodec2::TextCodec2(Context& ctx)
{
    const int blockSize = ctx.getInt("blockSize", 0);
    const int log = (blockSize >= 32) ? max(min(Global::log2(blockSize / 32), 24), 13) : 13;
    _logHashSize = (ctx.getInt("extra", 0) == 0) ? log : log + 1;
    _dictSize = 1 << 13;
    _dictMap = nullptr;
    _dictList = nullptr;
    _hashMask = (1 << _logHashSize) - 1;
    _staticDictSize = TextCodec::STATIC_DICT_WORDS;
    _isCRLF = false;
    _pCtx = &ctx;
}

void TextCodec2::reset(int count)
{
    // Select an appropriate initial dictionary size
    const int log = (count < 1024) ? 13 : max(min(Global::log2(count / 128), 18), 13);
    _dictSize = max(TextCodec::STATIC_DICT_WORDS, 1 << log);
    const int mapSize = 1 << _logHashSize;

    if (_dictMap == nullptr)
        _dictMap = new DictEntry*[mapSize];

    for (int i = 0; i < mapSize; i++)
        _dictMap[i] = nullptr;

    if (_dictList == nullptr) {
        _dictList = new DictEntry[_dictSize];
#if __cplusplus >= 201103L
        memcpy(&_dictList[0], &TextCodec::STATIC_DICTIONARY[0], sizeof(TextCodec::STATIC_DICTIONARY));
#else
	for (int i = 0; i < TextCodec::STATIC_DICT_WORDS; i++)
        _dictList[i] = TextCodec::STATIC_DICTIONARY[i];
#endif
    }

    for (int i = 0; i < _staticDictSize; i++)
        _dictMap[_dictList[i]._hash & _hashMask] = &_dictList[i];

    // Pre-allocate all dictionary entries
    for (int i = _staticDictSize; i < _dictSize; i++)
        _dictList[i] = DictEntry(nullptr, 0, i);
}

bool TextCodec2::forward(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    if (output._length - output._index < getMaxEncodedLength(count))
        return false;

    byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];

    if (_pCtx != nullptr) {
        Global::DataType dt = (Global::DataType) _pCtx->getInt("dataType", Global::UNDEFINED);

        // Filter out most types. Still check binaries which may contain significant parts of text
        if ((dt != Global::UNDEFINED) && (dt != Global::TEXT) && (dt != Global::BIN))
            return false;
    }

    uint freqs[256] = { 0 };
    byte mode = TextCodec::computeStats(&src[0], count, freqs, false);

    // Not text ?
    if ((mode & TextCodec::MASK_NOT_TEXT) != byte(0)) {
        if (_pCtx != nullptr)
            _pCtx->putInt("dataType", Global::DataType(mode & TextCodec::MASK_DT));

        return false;
    }

    if (_pCtx != nullptr)
       _pCtx->putInt("dataType", Global::TEXT);

    reset(count);
    const int srcEnd = count;
    const int dstEnd = getMaxEncodedLength(count);
    const int dstEnd3 = dstEnd - 3;
    int emitAnchor = 0; // never less than 0
    int words = _staticDictSize;

    // DOS encoded end of line (CR+LF) ?
    _isCRLF = (mode & TextCodec::MASK_CRLF) != byte(0);
    dst[0] = mode;
    bool res = true;
    int srcIdx = 0;
    int dstIdx = 1;

    while ((srcIdx < srcEnd) && (src[srcIdx] == TextCodec::SP)) {
        dst[dstIdx++] = TextCodec::SP;
        srcIdx++;
        emitAnchor++;
    }

    int delimAnchor = TextCodec::isText(src[srcIdx]) ? srcIdx - 1 : srcIdx; // previous delimiter

    while (srcIdx < srcEnd) {
        if (TextCodec::isText(src[srcIdx])) {
            srcIdx++;
            continue;
        }

        if ((srcIdx > delimAnchor + 2) && TextCodec::isDelimiter(src[srcIdx])) {
            const byte val = src[delimAnchor + 1];
            const int length = srcIdx - delimAnchor - 1;

            if (length <= TextCodec::MAX_WORD_LENGTH) {
                // Compute hashes
                // h1 -> hash of word chars
                // h2 -> hash of word chars with first char case flipped
                int h1 = TextCodec::HASH1;
                h1 = h1 * TextCodec::HASH1 ^ int(val) * TextCodec::HASH2;
                int h2 = TextCodec::HASH1;
                h2 = h2 * TextCodec::HASH1 ^ (int(val) ^ 0x20) * TextCodec::HASH2;

                for (int i = delimAnchor + 2; i < srcIdx; i++) {
                    h1 = h1 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;
                    h2 = h2 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;
                }

                // Check word in dictionary
                DictEntry* pe = nullptr;
                prefetchRead(&_dictMap[h1 & _hashMask]);
                DictEntry* pe1 = _dictMap[h1 & _hashMask];

                // Check for hash collisions
                if ((pe1 != nullptr) && (pe1->_hash == h1) && ((pe1->_data >> 24) == length))
                    pe = pe1;
                else {
                    prefetchRead(&_dictMap[h2 & _hashMask]);
                    DictEntry* pe2 = _dictMap[h2 & _hashMask];

                    if ((pe2 != nullptr) && (pe2->_hash == h2) && ((pe2->_data >> 24) == length))
                        pe = pe2;
                }

                if (pe != nullptr) {
                    if (!TextCodec::sameWords(&pe->_ptr[1], &src[delimAnchor + 2], length - 1))
                        pe = nullptr;
                }

                if (pe == nullptr) {
                    // Word not found in the dictionary or hash collision.
                    // Replace entry if not in static dictionary
                    if (((length > 3) || ((length == 3) && (words < TextCodec::THRESHOLD2))) && (pe1 == nullptr)) {
                        DictEntry* pe3 = &_dictList[words];

                        if ((pe3->_data & TextCodec::MASK_LENGTH) >= _staticDictSize) {
                            // Reuse old entry
                            _dictMap[pe3->_hash & _hashMask] = nullptr;
                            pe3->_ptr = &src[delimAnchor + 1];
                            pe3->_hash = h1;
                            pe3->_data = (length << 24) | words;
                        }

                        // Update hash map
                        _dictMap[h1 & _hashMask] = pe3;
                        words++;

                        // Dictionary full ? Expand or reset index to end of static dictionary
                        if (words >= _dictSize) {
                            if (expandDictionary() == false)
                                words = _staticDictSize;
                        }
                    }
                }
                else {
                    // Word found in the dictionary
                    // Skip space if only delimiter between 2 word references
                    if ((emitAnchor != delimAnchor) || (src[delimAnchor] != TextCodec::SP)) {
                        const int dIdx = emitSymbols(&src[emitAnchor], &dst[dstIdx], delimAnchor + 1 - emitAnchor, dstEnd - dstIdx);

                        if (dIdx < 0) {
                            res = false;
                            break;
                        }

                        dstIdx += dIdx;
                    }

                    if (dstIdx >= dstEnd3) {
                        res = false;
                        break;
                    }

                    dstIdx += emitWordIndex(&dst[dstIdx], pe->_data & TextCodec::MASK_LENGTH, (pe == pe1) ? 0 : 32);
                    emitAnchor = delimAnchor + 1 + (pe->_data >> 24);
                }
            }
        }

        // Reset delimiter position
        delimAnchor = srcIdx;
        srcIdx++;
    }

    if (res == true) {
        // Emit last symbols
        const int dIdx = emitSymbols(&src[emitAnchor], &dst[dstIdx], srcEnd - emitAnchor, dstEnd - dstIdx);

        if (dIdx < 0)
            res = false;
        else
            dstIdx += dIdx;

        res &= (srcIdx == srcEnd);
    }

    output._index += dstIdx;
    input._index += srcIdx;
    return res;
}

bool TextCodec2::expandDictionary()
{
    if (_dictSize >= TextCodec::MAX_DICT_SIZE)
        return false;

    DictEntry* newDict = new DictEntry[_dictSize * 2];
    memcpy(static_cast<void*>(&newDict[0]), &_dictList[0], sizeof(DictEntry) * _dictSize);

    for (int i = _dictSize; i < _dictSize * 2; i++)
        newDict[i] = DictEntry(nullptr, 0, i);

    delete[] _dictList;
    _dictList = newDict;

    // Reset map (values must point to addresses of new DictEntry items)
    for (int i = 0; i < _dictSize; i++) {
        _dictMap[_dictList[i]._hash & _hashMask] = &_dictList[i];
    }

    _dictSize <<= 1;
    return true;
}

int TextCodec2::emitSymbols(const byte src[], byte dst[], const int srcEnd, const int dstEnd)
{
// Work around incorrect warning by GCC 7.x.x with C++17
#ifdef __GNUC__
    #if (__GNUC__ == 7) && (__cplusplus > 201402L)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wswitch"
    #endif
#endif

    int dstIdx = 0;

    if (2 * srcEnd < dstEnd) {
        for (int i = 0; i < srcEnd; i++) {
            const byte cur = src[i];

            switch (cur) {
            case TextCodec::ESCAPE_TOKEN1:
                dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
                dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
                break;

            case TextCodec::CR:
                if (_isCRLF == false)
                    dst[dstIdx++] = cur;

                break;

            default:
                if (cur >= byte(128))
                    dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;

                dst[dstIdx++] = cur;
            }
        }
    }
    else {
        for (int i = 0; i < srcEnd; i++) {
            const byte cur = src[i];

            switch (cur) {
            case TextCodec::ESCAPE_TOKEN1:
                if (dstIdx >= dstEnd - 1)
                    return -1;

                dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
                dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
                break;

            case TextCodec::CR:
                if (_isCRLF == false) {
                    if (dstIdx >= dstEnd)
                        return -1;

                    dst[dstIdx++] = cur;
                }

                break;

            default:
                if (cur >= byte(128)) {
                    if (dstIdx >= dstEnd)
                        return -1;

                    dst[dstIdx++] = TextCodec::ESCAPE_TOKEN1;
                }

                if (dstIdx >= dstEnd)
                    return -1;

                dst[dstIdx++] = cur;
            }
        }
    }

// Work around incorrect warning by GCC 7.x.x with C++17
#ifdef __GNUC__
    #if (__GNUC__ == 7) && (__cplusplus > 201402L)
        #pragma GCC diagnostic pop
    #endif
#endif

    return dstIdx;
}

int TextCodec2::emitWordIndex(byte dst[], int val, int mask)
{
    // Emit word index (varint 5 bits + 7 bits + 7 bits)
    // 1st byte: 0x80 => word idx, 0x40 => more bytes, 0x20 => toggle case 1st symbol
    // 2nd byte: 0x80 => 1 more byte
    if (val >= TextCodec::THRESHOLD3) {
        if (val >= TextCodec::THRESHOLD4) {
            // 5 + 7 + 7 => 2^19
            dst[0] = byte(0xC0 | mask | ((val >> 14) & 0x1F));
            dst[1] = byte(0x80 | (val >> 7));
            dst[2] = byte(val & 0x7F);
            return 3;
        }

        // 5 + 7 => 2^12 = 32*128
        dst[0] = byte(0xC0 | mask | (val >> 7));
        dst[1] = byte(val & 0x7F);
        return 2;
    }

    // 5 => 32
    dst[0] = byte(0x80 | mask | val);
    return 1;
}

bool TextCodec2::inverse(SliceArray<byte>& input, SliceArray<byte>& output, int count)
{
    reset(output._length);
    byte* src = &input._array[input._index];
    byte* dst = &output._array[output._index];
    _isCRLF = (src[0] & TextCodec::MASK_CRLF) != byte(0);
    int srcIdx = 1;
    int dstIdx = 0;
    const int srcEnd = count;
    const int dstEnd = output._length;
    int delimAnchor = TextCodec::isText(src[srcIdx]) ? srcIdx - 1 : srcIdx; // previous delimiter
    int words = _staticDictSize;
    bool wordRun = false;

    while ((srcIdx < srcEnd) && (dstIdx < dstEnd)) {
        const byte cur = src[srcIdx];

        if (TextCodec::isText(cur)) {
            dst[dstIdx] = cur;
            srcIdx++;
            dstIdx++;
            continue;
        }

        if ((srcIdx > delimAnchor + 3) && TextCodec::isDelimiter(cur)) {
            const int length = srcIdx - delimAnchor - 1; // length > 2

            if (length <= TextCodec::MAX_WORD_LENGTH) {
                int h1 = TextCodec::HASH1;

                for (int i = delimAnchor + 1; i < srcIdx; i++)
                    h1 = h1 * TextCodec::HASH1 ^ int(src[i]) * TextCodec::HASH2;

                // Lookup word in dictionary
                DictEntry* pe = nullptr;
                DictEntry* pe1 = _dictMap[h1 & _hashMask];

                // Check for hash collisions
                if ((pe1 != nullptr) && (pe1->_hash == h1) && ((pe1->_data >> 24) == length)) {
                    if (TextCodec::sameWords(&pe1->_ptr[1], &src[delimAnchor + 2], length - 1))
                        pe = pe1;
                }

                if (pe == nullptr) {
                    // Word not found in the dictionary or hash collision.
                    // Replace entry if not in static dictionary
                    if (((length > 3) || (words < TextCodec::THRESHOLD2)) && (pe1 == nullptr)) {
                        DictEntry& e = _dictList[words];

                        if ((e._data & TextCodec::MASK_LENGTH) >= _staticDictSize) {
                            // Reuse old entry
                            _dictMap[e._hash & _hashMask] = nullptr;
                            e._ptr = &src[delimAnchor + 1];
                            e._hash = h1;
                            e._data = (length << 24) | words;
                        }

                        _dictMap[h1 & _hashMask] = &e;
                        words++;

                        // Dictionary full ? Expand or reset index to end of static dictionary
                        if (words >= _dictSize) {
                            if (expandDictionary() == false)
                                words = _staticDictSize;
                        }
                    }
                }
            }
        }

        srcIdx++;

        if (cur >= TextCodec::MASK_80) {
            // Word in dictionary
            // Read word index (varint 5 bits + 7 bits + 7 bits)
            int idx = int(cur & TextCodec::MASK_1F);

            if ((cur & TextCodec::MASK_40) != byte(0)) {
                const int idx2 = int(src[srcIdx++]);

                if (idx2 >= 128) {
                    idx = (idx << 14) | ((idx2 & 0x7F) << 7) | int(src[srcIdx]);
                    srcIdx++;
                }
                else {
                    idx = (idx << 7) | idx2;
		}

                if (idx >= _dictSize)
                    break;
            }

            const int length = _dictList[idx]._data >> 24;

            // Sanity check
            if (dstIdx + length >= dstEnd)
                break;

            // Emit word
            if (length > 1) {
                // Add space if only delimiter between 2 words (not an escaped delimiter)
                if (wordRun == true)
                    dst[dstIdx++] = TextCodec::SP;

                // Regular word entry
                wordRun = true;
                delimAnchor = srcIdx;
            }
            else {
                if (length == 0)
                   break;

                // Escape entry
                wordRun = false;
                delimAnchor = srcIdx - 1;
            }

            memcpy(&dst[dstIdx], _dictList[idx]._ptr, length);

            // Flip case of first character ?
            dst[dstIdx] ^= (cur & TextCodec::MASK_20);
            dstIdx += length;
        }
        else {
            if (cur == TextCodec::ESCAPE_TOKEN1) {
                dst[dstIdx++] = src[srcIdx++];
            }
            else {
                if ((_isCRLF == true) && (cur == TextCodec::LF))
                    dst[dstIdx++] = TextCodec::CR;

                dst[dstIdx++] = cur;
            }

            wordRun = false;
            delimAnchor = srcIdx - 1;
        }
    }

    output._index += dstIdx;
    input._index += srcIdx;
    return srcIdx == srcEnd;
}
