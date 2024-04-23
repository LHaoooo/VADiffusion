/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2017, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TEncSearch.h
    \brief    encoder search class (header)
*/

#ifndef __TENCSEARCH__
#define __TENCSEARCH__

// Include files
#include "TLibCommon/TComYuv.h"
#include "TLibCommon/TComMotionInfo.h"
#include "TLibCommon/TComPattern.h"
#include "TLibCommon/TComPrediction.h"
#include "TLibCommon/TComTrQuant.h"
#include "TLibCommon/TComPic.h"
#include "TLibCommon/TComRectangle.h"
#include "TEncEntropy.h"
#include "TEncSbac.h"
#include "TEncCfg.h"
#include "TLibCommon/TComHash.h"

//! \ingroup TLibEncoder
//! \{

class TEncCu;

struct IntraBCHashNode
{
  Int posX;
  Int posY;
  IntraBCHashNode * next;
};

// ====================================================================================================================
// Class definition
// ====================================================================================================================

static const UInt MAX_NUM_REF_LIST_ADAPT_SR=2;
static const UInt MAX_IDX_ADAPT_SR=33;
static const UInt NUM_MV_PREDICTORS=3;

/// encoder search class
class TEncSearch : public TComPrediction
{
private:
  TCoeff**        m_ppcQTTempCoeff[MAX_NUM_COMPONENT /* 0->Y, 1->Cb, 2->Cr*/];
#if ADAPTIVE_QP_SELECTION
  TCoeff**        m_ppcQTTempArlCoeff[MAX_NUM_COMPONENT];
#endif
  UChar*          m_puhQTTempTrIdx;
  UChar*          m_puhQTTempCbf[MAX_NUM_COMPONENT];

  TComYuv*        m_pcQTTempTComYuv;
  TComYuv         m_tmpYuvPred; // To be used in xGetInterPredictionError() to avoid constant memory allocation/deallocation

  SChar*          m_phQTTempCrossComponentPredictionAlpha[MAX_NUM_COMPONENT];
  Pel*            m_pSharedPredTransformSkip[MAX_NUM_COMPONENT];
  TCoeff*         m_pcQTTempTUCoeff[MAX_NUM_COMPONENT];
  UChar*          m_puhQTTempTransformSkipFlag[MAX_NUM_COMPONENT];
  TComYuv         m_pcQTTempTransformSkipTComYuv;
#if ADAPTIVE_QP_SELECTION
  TCoeff*         m_ppcQTTempTUArlCoeff[MAX_NUM_COMPONENT];
#endif

protected:
  // interface to option
  TEncCfg*        m_pcEncCfg;

  // interface to classes
  TComTrQuant*    m_pcTrQuant;
  TComRdCost*     m_pcRdCost;
  TEncEntropy*    m_pcEntropyCoder;

  // ME parameters
  Int             m_iSearchRange;
  Int             m_bipredSearchRange; // Search range for bi-prediction
  MESearchMethod  m_motionEstimationSearchMethod;
  Int             m_aaiAdaptSR[MAX_NUM_REF_LIST_ADAPT_SR][MAX_IDX_ADAPT_SR];
  TComMv          m_acMvPredictors[NUM_MV_PREDICTORS]; // Left, Above, AboveRight. enum MVP_DIR first NUM_MV_PREDICTORS entries are suitable for accessing.

  // RD computation
  TEncSbac***     m_pppcRDSbacCoder;
  TEncSbac*       m_pcRDGoOnSbacCoder;
  DistParam       m_cDistParam;

  // Misc.
  Pel*            m_pTempPel;

  // AMVP cost computation
  // UInt            m_auiMVPIdxCost[AMVP_MAX_NUM_CANDS+1][AMVP_MAX_NUM_CANDS];
  UInt            m_auiMVPIdxCost[AMVP_MAX_NUM_CANDS+1][AMVP_MAX_NUM_CANDS+1]; //th array bounds

  TComMv          m_integerMv2Nx2N[NUM_REF_PIC_LIST_01][MAX_NUM_REF];

  Bool            m_isInitialized;

  IntraBCHashNode***      m_pcIntraBCHashTable;                 ///< The hash table used for Intra BC search
  Bool*           m_puhQTTempACTFlag;
  TCoeff*         m_pcACTTempTUCoeff[MAX_NUM_COMPONENT];
  TComYuv         m_pcACTTempTransformSkipTComYuv;
  TComYuv*        m_pcQTTempTComYuvCS;
  TComYuv*        m_pcNoCorrYuvTmp;
#if ADAPTIVE_QP_SELECTION
  TCoeff*         m_ppcACTTempTUArlCoeff[MAX_NUM_COMPONENT];
#endif
  TComACTTURDCost m_sACTRDCostTU[5];
  RefPicList      m_currRefPicList;
  Int             m_currRefPicIndex;
  Bool            m_bSkipFracME;
  TComMv          m_acBVs[SCM_S0067_NUM_CANDIDATES];
  UInt            m_numBVs, m_numBV16s;
  Distortion      m_lastCandCost;
  Bool            m_bBestScanRotationMode;
  Pel*            m_paOriginalLevel;
  Pel*            m_paBestLevel[MAX_NUM_COMPONENT];
  UChar*          m_paBestSPoint;
  TCoeff*         m_paBestRun;
  UChar*          m_paBestEscapeFlag;
  UInt            m_prevPaletteSize[MAX_PALETTE_ITER];
  Pel             m_prevPalette[MAX_PALETTE_ITER][3][MAX_PALETTE_SIZE];
  UInt            m_forcePaletteSize;
  Pel             m_forcePalette[3][MAX_PALETTE_SIZE];

  Pel*            m_paLevelStoreRD[MAX_NUM_COMPONENT];
  UChar*          m_paSPointStoreRD;
  TCoeff*         m_paRunStoreRD;
  UChar*          m_paEscapeFlagStoreRD;

  UInt m_runGolombGroups[32*32];

  PaletteInfoStruct m_currentPaletteElement;
  PaletteInfoStruct m_nextPaletteElement;

  Int               m_paletteErrLimit;
  Pel               m_indexBlock[MAX_CU_SIZE * MAX_CU_SIZE];
  UInt              m_indError[32*32][MAX_PALETTE_SIZE+1];
  UInt              m_paletteNoElements;

  PaletteInfoStruct m_paletteInfo[32*32];
  PaletteInfoStruct m_paletteInfoBest[32*32];
  PaletteInfoStruct m_paletteInfoStoreRD[32*32];

  UInt              m_paletteNoElementsBest;
  Pel               m_posBlock[32*32];
  Pel               m_posBlockRD[32*32];
  Pel               m_indexBlockRD[MAX_CU_SIZE * MAX_CU_SIZE];

  UInt              m_maxVal[3];
  Int               m_quantiserScale[3];
  Int               m_quantiserRightShift[3];
  Int               m_rightShiftOffset[3];
  Int               m_invQuantScales[3];
  Int               m_qpPer[3];
  UShort**          m_truncBinBits; //ZF two dimensions related with input bitdepth
  UShort*           m_escapeNumBins; // number of binarization bins fpr escape pixels
  UInt              m_maxSymbolSize;
  UInt              m_symbolSize;

  UChar*            m_paletteRunMode;
  UChar*            m_escapeFlagRD;
  TCoeff*           m_runRD;
  Pel*              m_levelRD[MAX_NUM_COMPONENT];

public:
  TEncSearch();
  virtual ~TEncSearch();

  Void init(TEncCfg*       pcEncCfg,
            TComTrQuant*   pcTrQuant,
            Int            iSearchRange,
            Int            bipredSearchRange,
            MESearchMethod motionEstimationSearchMethod,
            const UInt     maxCUWidth,
            const UInt     maxCUHeight,
            const UInt     maxTotalCUDepth,
            TEncEntropy*   pcEntropyCoder,
            TComRdCost*    pcRdCost,
            TEncSbac***    pppcRDSbacCoder,
            TEncSbac*      pcRDGoOnSbacCoder );

  Void destroy();

protected:

  /// sub-function for motion vector refinement used in fractional-pel accuracy
  Distortion  xPatternRefinement( TComPattern* pcPatternKey,
                                  TComMv baseRefMv,
                                  Int iFrac, TComMv& rcMvFrac, Bool bAllowUseOfHadamard
                                 );

  typedef struct
  {
    const Pel*  piRefY;
    Int         iYStride;
    Int         iBestX;
    Int         iBestY;
    UInt        uiBestRound;
    UInt        uiBestDistance;
    Distortion  uiBestSad;
    UChar       ucPointNr;
  } IntTZSearchStruct;

  // sub-functions for ME
  __inline Void xTZSearchHelp         ( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const Int iSearchX, const Int iSearchY, const UChar ucPointNr, const UInt uiDistance );
  __inline Void xTZ2PointSearch       ( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const TComMv* const pcMvSrchRngLT, const TComMv* const pcMvSrchRngRB );
  __inline Void xTZ8PointSquareSearch ( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const TComMv* const pcMvSrchRngLT, const TComMv* const pcMvSrchRngRB, const Int iStartX, const Int iStartY, const Int iDist );
  __inline Void xTZ8PointDiamondSearch( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const TComMv* const pcMvSrchRngLT, const TComMv* const pcMvSrchRngRB, const Int iStartX, const Int iStartY, const Int iDist, const Bool bCheckCornersAtDist1, Bool bSkipLeftDist2 = false, Bool bSkipTopDist2 = false  );

  Void xGetInterPredictionError( TComDataCU* pcCU, TComYuv* pcYuvOrg, Int iPartIdx, Distortion& ruiSAD, Bool Hadamard );

public:
  Void  estIntraPredLumaQT      ( TComDataCU* pcCU,
                                  TComYuv*    pcOrgYuv,
                                  TComYuv*    pcPredYuv,
                                  TComYuv*    pcResiYuv,
                                  TComYuv*    pcRecoYuv,
                                  Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                  DEBUG_STRING_FN_DECLARE(sDebug));

  Void  estIntraPredChromaQT    ( TComDataCU* pcCU,
                                  TComYuv*    pcOrgYuv,
                                  TComYuv*    pcPredYuv,
                                  TComYuv*    pcResiYuv,
                                  TComYuv*    pcRecoYuv,
                                  Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                  DEBUG_STRING_FN_DECLARE(sDebug));

  /// encoder estimation - inter prediction (non-skip)
  Bool predInterSearch          ( TComDataCU* pcCU,
                                  TComYuv*    pcOrgYuv,
                                  TComYuv*    pcPredYuv,
                                  TComYuv*    pcResiYuv,
                                  TComYuv*    pcRecoYuv
                                  DEBUG_STRING_FN_DECLARE(sDebug),
                                  Bool        bUseRes = false
#if AMP_MRG
                                 ,Bool        bUseMRG = false
#endif
                                , TComMv*     iMVCandList = NULL
                                );

  /// encode residual and compute rd-cost for inter mode
  Void encodeResAndCalcRdInterCU( TComDataCU* pcCU,
                                  TComYuv*    pcYuvOrg,
                                  TComYuv*    pcYuvPred,
                                  TComYuv*    pcYuvResi,
                                  TComYuv*    pcYuvResiBest,
                                  TComYuv*    pcYuvRec,
                                  Bool        bSkipResidual,
                                  TComYuv*    pcYuvNoCorrResi,
                                  ACTRDTestTypes eACTRDTestType
                                  DEBUG_STRING_FN_DECLARE(sDebug) );

  /// set ME search range
  Void setAdaptiveSearchRange   ( Int iDir, Int iRefIdx, Int iSearchRange) { assert(iDir < MAX_NUM_REF_LIST_ADAPT_SR && iRefIdx<Int(MAX_IDX_ADAPT_SR)); m_aaiAdaptSR[iDir][iRefIdx] = iSearchRange; }

  Void xEncPCM    (TComDataCU* pcCU, UInt uiAbsPartIdx, Pel* piOrg, Pel* piPCM, Pel* piPred, Pel* piResi, Pel* piReco, UInt uiStride, UInt uiWidth, UInt uiHeight, const ComponentID compID );
  Void IPCMSearch (TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* rpcPredYuv, TComYuv* rpcResiYuv, TComYuv* rpcRecoYuv );

  Void  estIntraPredQTCT        ( TComDataCU*    pcCU,
                                  TComYuv*       pcOrgYuv,
                                  TComYuv*       pcPredYuv,
                                  TComYuv*       pcResiYuv,
                                  TComYuv*       pcRecoYuv,
                                  ACTRDTestTypes eACTRDTestType,
                                  Bool           bReuseIntraMode
                                  DEBUG_STRING_FN_DECLARE(sDebug)
                                 );

  Void  estIntraPredLumaQTWithModeReuse ( TComDataCU* pcCU,
                                          TComYuv*    pcOrgYuv,
                                          TComYuv*    pcPredYuv,
                                          TComYuv*    pcResiYuv,
                                          TComYuv*    pcRecoYuv,
                                          Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                    );

  Void  estIntraPredChromaQTWithModeReuse ( TComDataCU* pcCU,
                                            TComYuv*    pcOrgYuv,
                                            TComYuv*    pcPredYuv,
                                            TComYuv*    pcResiYuv,
                                            TComYuv*    pcRecoYuv,
                                            Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                          );

  Bool isBlockVectorValid( Int xPos, Int yPos, Int width, Int height, TComDataCU *pcCU,
                           Int xStartInCU, Int yStartInCU, Int xBv, Int yBv, Int ctuSize );

  Bool predIntraBCSearch        ( TComDataCU* pcCU,
                                  TComYuv*    pcOrgYuv,
                                  TComYuv*&   rpcPredYuv,
                                  TComYuv*&   rpcResiYuv,
                                  TComYuv*&   rpcRecoYuv
                                  DEBUG_STRING_FN_DECLARE(sDebug),
                                  Bool        bUse1DSearchFor8x8,
                                  Bool        bUseRes,
                                  Bool        testOnlyPred
                                );

  Bool predMixedIntraBCInterSearch( TComDataCU* pcCU,
                                    TComYuv*    pcOrgYuv,
                                    TComYuv*&   rpcPredYuv,
                                    TComYuv*&   rpcResiYuv,
                                    TComYuv*&   rpcRecoYuv
                                    DEBUG_STRING_FN_DECLARE( sDebug ),
                                    TComMv*     iMVCandList,
                                    Bool        bUseRes
                                    );

  Void xIntraBlockCopyEstimation( TComDataCU*  pcCU,
                                  TComYuv*     pcYuvOrg,
                                  Int          iPartIdx,
                                  TComMv*      pcMvPred,
                                  TComMv&      rcMv,
                                  Distortion&  cost,
                                  Bool         bUse1DSearchFor8x8,
                                  Bool         testOnlyPred
                                );

  Void addToSortList            ( list<BlockHash>& listBlockHash,
                                  list<Int>& listCost,
                                  Int cost,
                                  const BlockHash& blockHash
                                );

  Distortion getSAD             ( Pel* pRef,
                                  Int refStride,
                                  Pel* pCurr,
                                  Int currStride,
                                  Int width,
                                  Int height,
                                  const BitDepths& bitDepths
                                );

  Bool predInterHashSearch      ( TComDataCU* pcCU,
                                  TComYuv*& rpcPredYuv,
                                  Bool& isPerfectMatch
                                );

  Bool xHashInterEstimation     ( TComDataCU* pcCU,
                                  Int width,
                                  Int height,
                                  RefPicList& bestRefPicList,
                                  Int& bestRefIndex,
                                  TComMv& bestMv,
                                  TComMv& bestMvd,
                                  Int& bestMVPIndex,
                                  Bool& isPerfectMatch
                                 );

  Int  xHashInterPredME         ( const TComDataCU* const pcCU,
                                  Int width,
                                  Int height,
                                  RefPicList currRefPicList,
                                  Int currRefPicIndex,
                                  TComMv bestMv[5]
                                );

  Void selectMatchesInter       ( const MapIterator& itBegin,
                                  Int count,
                                  list<BlockHash>& vecBlockHash,
                                  const BlockHash& currBlockHash
                                 );

  Void xSetIntraSearchRange     ( TComDataCU*   pcCU,
                                  TComMv&       cMvPred,
                                  UInt          partAddr,
                                  Int           roiWidth,
                                  Int           roiHeight,
                                  TComMv&       rcMvSrchRngLT,
                                  TComMv&       rcMvSrchRngRB );


  Bool isValidIntraBCSearchArea(  TComDataCU*   pcCU,
                                  Int           predX,
                                  Int           predY,
                                  Int           roiWidth,
                                  Int           roiHeight,
                                  Int           uiPartOffset)
  {
    const Int  cuPelX        = pcCU->getCUPelX() + g_auiRasterToPelX[ g_auiZscanToRaster[ uiPartOffset ] ];
    const Int  cuPelY        = pcCU->getCUPelY() + g_auiRasterToPelY[ g_auiZscanToRaster[ uiPartOffset ] ];

    if ( !isBlockVectorValid(cuPelX,cuPelY,roiWidth,roiHeight,pcCU,g_auiRasterToPelX[ g_auiZscanToRaster[ uiPartOffset ] ],g_auiRasterToPelY[ g_auiZscanToRaster[ uiPartOffset ] ],predX,predY,pcCU->getSlice()->getSPS()->getMaxCUWidth()))
    {
      return false;
    }

    const Int uiMaxCuWidth   = pcCU->getSlice()->getSPS()->getMaxCUWidth();
    const Int uiMaxCuHeight  = pcCU->getSlice()->getSPS()->getMaxCUHeight();
    const UInt curTileIdx = pcCU->getPic()->getPicSym()->getTileIdxMap( pcCU->getCtuRsAddr() );
    TComTile* curTile = pcCU->getPic()->getPicSym()->getTComTile( curTileIdx );

    const Int tileAreaRight  = (curTile->getRightEdgePosInCtus() + 1) * uiMaxCuWidth;
    const Int tileAreaBottom = (curTile->getBottomEdgePosInCtus() + 1) * uiMaxCuHeight;

    const Int tileAreaLeft   = tileAreaRight - curTile->getTileWidthInCtus() * uiMaxCuWidth;
    const Int tileAreaTop    = tileAreaBottom - curTile->getTileHeightInCtus() * uiMaxCuHeight;

    if( (cuPelX + predX + roiWidth > tileAreaRight) || (cuPelY + predY + roiHeight > tileAreaBottom) ||
      (cuPelX + predX < tileAreaLeft) || (cuPelY + predY < tileAreaTop) )
    {
      return false;
    }

    TComSlice *pcSlice = pcCU->getSlice();
    if( pcSlice->getSliceMode() )
    {
      TComPicSym *pcSym = pcCU->getPic()->getPicSym();
      Int      ctuX = (cuPelX + predX) / uiMaxCuWidth;
      Int      ctuY = (cuPelY + predY) / uiMaxCuHeight;
      UInt   refCtu = ctuX + pcSym->getFrameWidthInCtus()*ctuY;
      UInt startCtu = pcSym->getCtuTsToRsAddrMap( pcCU->getSlice()->getSliceSegmentCurStartCtuTsAddr() );
      if (refCtu < startCtu) return false;
    }
    return true;
  }

  Void xIntraBCSearchMVCandUpdate(Distortion sad, Int x, Int y, Distortion* sadBestCand, TComMv* cMVCand);

  Int xIntraBCSearchMVChromaRefine( TComDataCU *pcCU,
                                    Int         roiWidth,
                                    Int         roiHeight,
                                    Int         cuPelX,
                                    Int         cuPelY,
                                    Distortion* sadBestCand,
                                    TComMv*     cMVCand,
                                    UInt        partAddr,
                                    Int         partIdx
                                    );

  Void xIntraPatternSearch      ( TComDataCU*  pcCU,
                                  Int          partIdx,
                                  UInt         partAddr,
                                  TComPattern* pcPatternKey,
                                  Pel*         piRefY,
                                  Int          refStride,
                                  TComMv*      pcMvSrchRngLT,
                                  TComMv*      pcMvSrchRngRB,
                                  TComMv&      rcMv,
                                  Distortion&  SAD,
                                  Int          roiWidth,
                                  Int          roiHeight,
                                  Bool         bUse1DSearchFor8x8,
                                  Bool         testOnlyPred
                                );

  Void updateBVMergeCandLists(int roiWidth, int roiHeight, TComMv* mvCand);

  UInt paletteSearch  (TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv*& rpcPredYuv, TComYuv*& rpcResiYuv, TComYuv*& rpcRecoYuv, Bool forcePalettePrediction, UInt iterNumber, UInt *paletteSize);

  Void deriveRunAndCalcBits( TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcRecoYuv, UInt& minBits, Bool bReset, PaletteScanMode paletteScanMode);

  Int xIntraBCHashTableIndex  ( TComDataCU* pcCU,
                                Int posX,
                                Int posY,
                                Int width,
                                Int height,
                                Bool isRec
                              );

  Void xIntraBCHashSearch     ( TComDataCU* pcCU,
                                TComYuv* pcYuvOrg,
                                Int partIdx,
                                TComMv* pcMvPred,
                                TComMv& rcMv,
                                UInt intraBCECost
                              );

  Void xIntraBCHashTableUpdate( TComDataCU* pcCU,
                                Bool isRec
                              );

  Void xClearIntraBCHashTable();

  Void setHashLinklist        ( IntraBCHashNode*& hashLinklist,
                                UInt depth,
                                UInt hashIdx
                              );

  IntraBCHashNode* getHashLinklist( UInt depth, Int hashIdx ) { return m_pcIntraBCHashTable[depth][hashIdx]; }

  Void  setPaletteErrLimit                ( Int paletteErrLimit )     { m_paletteErrLimit = paletteErrLimit; }
protected:

  // -------------------------------------------------------------------------------------------------------------------
  // Intra search
  // -------------------------------------------------------------------------------------------------------------------

  Void  xEncSubdivCbfQT           ( TComTU      &rTu,
                                    Bool         bLuma,
                                    Bool         bChroma );

  Void  xEncCoeffQT               ( TComTU &rTu,
                                    ComponentID  component,
                                    Bool         bRealCoeff );
  Void  xEncIntraHeader           ( TComDataCU*  pcCU,
                                    UInt         uiTrDepth,
                                    UInt         uiAbsPartIdx,
                                    Bool         bLuma,
                                    Bool         bChroma );
  UInt  xGetIntraBitsQT           ( TComTU &rTu,
                                    Bool         bLuma,
                                    Bool         bChroma,
                                    Bool         bRealCoeff );

  UInt  xGetIntraBitsQTChroma    ( TComTU &rTu,
                                   ComponentID compID,
                                   Bool          bRealCoeff );

  Void  xIntraCodingTUBlock       (       TComYuv*      pcOrgYuv,
                                          TComYuv*      pcPredYuv,
                                          TComYuv*      pcResiYuv,
                                          Pel           resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                    const Bool          checkCrossCPrediction,
                                          Distortion&   ruiDist,
                                    const ComponentID   compID,
                                          TComTU        &rTu
                                    DEBUG_STRING_FN_DECLARE(sTest)
                                         ,Int           default0Save1Load2 = 0
                                   );

  Void  xRecurIntraCodingLumaQT   ( TComYuv*    pcOrgYuv,
                                    TComYuv*    pcPredYuv,
                                    TComYuv*    pcResiYuv,
                                    Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                    Distortion& ruiDistY,
#if HHI_RQT_INTRA_SPEEDUP
                                    Bool         bCheckFirst,
#endif
                                    Double&      dRDCost,
                                    TComTU      &rTu
                                    DEBUG_STRING_FN_DECLARE(sDebug));

  Void  xSetIntraResultLumaQT     ( TComYuv*     pcRecoYuv,
                                    TComTU &rTu);

  Void xStoreCrossComponentPredictionResult  (       Pel    *pResiLuma,
                                               const Pel    *pBestLuma,
                                                     TComTU &rTu,
                                               const Int     xOffset,
                                               const Int     yOffset,
                                               const Int     strideResi,
                                               const Int     strideBest );

  SChar xCalcCrossComponentPredictionAlpha   (       TComTU &rTu,
                                               const ComponentID compID,
                                               const Pel*        piResiL,
                                               const Pel*        piResiC,
                                               const Int         width,
                                               const Int         height,
                                               const Int         strideL,
                                               const Int         strideC );

  Void  xRecurIntraChromaCodingQT ( TComYuv*    pcOrgYuv,
                                    TComYuv*    pcPredYuv,
                                    TComYuv*    pcResiYuv,
                                    Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                    Distortion& ruiDist,
                                    TComTU      &rTu
                                    DEBUG_STRING_FN_DECLARE(sDebug));

  Void  xSetIntraResultChromaQT   ( TComYuv*    pcRecoYuv, TComTU &rTu);

  Void  xStoreIntraResultQT       ( const ComponentID compID, TComTU &rTu, Bool bACTCache = false);
  Void  xLoadIntraResultQT        ( const ComponentID compID, TComTU &rTu, Bool bACTCache = false);


  // -------------------------------------------------------------------------------------------------------------------
  // Inter search (AMP)
  // -------------------------------------------------------------------------------------------------------------------

  Void xEstimateMvPredAMVP        ( TComDataCU* pcCU,
                                    TComYuv*    pcOrgYuv,
                                    UInt        uiPartIdx,
                                    RefPicList  eRefPicList,
                                    Int         iRefIdx,
                                    TComMv&     rcMvPred,
                                    Bool        bFilled = false
                                  , Distortion* puiDistBiP = NULL
                                     );

  Void xCheckBestMVP              ( TComDataCU* pcCU,
                                    RefPicList  eRefPicList,
                                    TComMv      cMv,
                                    TComMv&     rcMvPred,
                                    Int&        riMVPIdx,
                                    UInt&       ruiBits,
                                    Distortion& ruiCost );

  Distortion xGetTemplateCost    ( TComDataCU*  pcCU,
                                    UInt        uiPartAddr,
                                    TComYuv*    pcOrgYuv,
                                    TComYuv*    pcTemplateCand,
                                    TComMv      cMvCand,
                                    Int         iMVPIdx,
                                    Int         iMVPNum,
                                    RefPicList  eRefPicList,
                                    Int         iRefIdx,
                                    Int         iSizeX,
                                    Int         iSizeY
                                   );


  Void xCopyAMVPInfo              ( AMVPInfo*   pSrc, AMVPInfo* pDst );
  UInt xGetMvpIdxBits             ( Int iIdx, Int iNum );
  Void xGetBlkBits                ( PartSize  eCUMode, Bool bPSlice, Int iPartIdx,  UInt uiLastMode, UInt uiBlkBit[3]);

  Void xMergeEstimation           ( TComDataCU*  pcCU,
                                    TComYuv*     pcYuvOrg,
                                    Int          iPartIdx,
                                    UInt&        uiInterDir,
                                    TComMvField* pacMvField,
                                    UInt&        uiMergeIndex,
                                    Distortion&  ruiCost,
                                    TComMvField* cMvFieldNeighbours,
                                    UChar*       uhInterDirNeighbours,
                                    Int&         numValidMergeCand,
                                    Int          iCostCalcType = 0
                                   );

  Void xRestrictBipredMergeCand   ( TComDataCU*     pcCU,
                                    UInt            puIdx,
                                    TComMvField*    mvFieldNeighbours,
                                    UChar*          interDirNeighbours,
                                    Int             numValidMergeCand );


  // -------------------------------------------------------------------------------------------------------------------
  // motion estimation
  // -------------------------------------------------------------------------------------------------------------------

  Void xMotionEstimation          ( TComDataCU*  pcCU,
                                    TComYuv*     pcYuvOrg,
                                    Int          iPartIdx,
                                    RefPicList   eRefPicList,
                                    TComMv*      pcMvPred,
                                    Int          iRefIdxPred,
                                    TComMv&      rcMv,
                                    UInt&        ruiBits,
                                    Distortion&  ruiCost,
                                    Bool         bBi = false  );

  Void xTZSearch                  ( const TComDataCU* const  pcCU,
                                    const TComPattern* const pcPatternKey,
                                    const Pel* const         piRefY,
                                    const Int                iRefStride,
                                    const TComMv* const      pcMvSrchRngLT,
                                    const TComMv* const      pcMvSrchRngRB,
                                    TComMv&                  rcMv,
                                    Distortion&              ruiSAD,
                                    const TComMv* const      pIntegerMv2Nx2NPred,
                                    const Bool               bExtendedSettings
                                    );

  Void xTZSearchSelective         ( const TComDataCU* const  pcCU,
                                    const TComPattern* const pcPatternKey,
                                    const Pel* const         piRefY,
                                    const Int                iRefStride,
                                    const TComMv* const      pcMvSrchRngLT,
                                    const TComMv* const      pcMvSrchRngRB,
                                    TComMv&                  rcMv,
                                    Distortion&              ruiSAD,
                                    const TComMv* const      pIntegerMv2Nx2NPred
                                    );

  Void xSetSearchRange            ( const TComDataCU* const pcCU,
                                    const TComMv&      cMvPred,
                                    const Int          iSrchRng,
                                    TComMv&      rcMvSrchRngLT,
#if !MCTS_ENC_CHECK
                                    TComMv&      rcMvSrchRngRB );
#else
                                    TComMv&      rcMvSrchRngRB,
                                    const TComPattern* const pcPatternKey );
#endif

#if MCTS_ENC_CHECK
  Void xInitTileBorders(const TComDataCU* const pcCU, TComPattern* pcPatternKey);
#endif

  Void xPatternSearchFast         ( const TComDataCU* const  pcCU,
                                    const TComPattern* const pcPatternKey,
                                    const Pel* const         piRefY,
                                    const Int                iRefStride,
                                    const TComMv* const      pcMvSrchRngLT,
                                    const TComMv* const      pcMvSrchRngRB,
                                    TComMv&                  rcMv,
                                    Distortion&              ruiSAD,
                                    const TComMv* const      pIntegerMv2Nx2NPred
                                  );

  Void xPatternSearch             ( const TComPattern* const pcPatternKey,
                                    const Pel*               piRefY,
                                    const Int                iRefStride,
                                    const TComMv* const      pcMvSrchRngLT,
                                    const TComMv* const      pcMvSrchRngRB,
                                    TComMv&      rcMv,
                                    Distortion&  ruiSAD );

  Void xPatternSearchFracDIF      (
                                    Bool         bIsLosslessCoded,
                                    TComDataCU*  pcCU,
                                    TComPattern* pcPatternKey,
                                    Pel*         piRefY,
                                    Int          iRefStride,
                                    TComMv*      pcMvInt,
                                    TComMv&      rcMvHalf,
                                    TComMv&      rcMvQter,
                                    Distortion&  ruiCost
                                   );

  Void xExtDIFUpSamplingH( TComPattern* pcPattern );
  Void xExtDIFUpSamplingQ( TComPattern* pcPatternKey, TComMv halfPelRef );

  // -------------------------------------------------------------------------------------------------------------------
  // T & Q & Q-1 & T-1
  // -------------------------------------------------------------------------------------------------------------------


  Void xEncodeInterResidualQT( const ComponentID compID, TComTU &rTu );
  Void xEstimateInterResidualQT( TComYuv* pcResi, Double &rdCost, UInt &ruiBits, Distortion &ruiDist, Distortion *puiZeroDist, TComTU &rTu DEBUG_STRING_FN_DECLARE(sDebug), TComYuv* pcOrgResi = NULL );
  Void xSetInterResidualQTData( TComYuv* pcResi, Bool bSpatial, TComTU &rTu  );

  UInt  xModeBitsIntra ( TComDataCU* pcCU, UInt uiMode, UInt uiPartOffset, UInt uiDepth, const ChannelType compID );
  UInt  xUpdateCandList( UInt uiMode, Double uiCost, UInt uiFastCandNum, UInt * CandModeList, Double * CandCostList );

  // -------------------------------------------------------------------------------------------------------------------
  // compute symbol bits
  // -------------------------------------------------------------------------------------------------------------------

  Void xAddSymbolBitsInter       ( TComDataCU*   pcCU,
                                   UInt&         ruiBits);

  Void  setWpScalingDistParam( TComDataCU* pcCU, Int iRefIdx, RefPicList eRefPicListCur );
  inline  Void  setDistParamComp( ComponentID compIdx )  { m_cDistParam.compIdx = compIdx; }

  // SCM new added functions
  TComYuv* getTmpYuvPred() {return &m_tmpYuvPred;}
  Void  xEncColorTransformFlagQT  ( TComTU &rTu );

  Void  xIntraCodingTUBlockCSC    (       TComYuv*      pcResiYuv,
                                          Pel           resiLuma[MAX_CU_SIZE * MAX_CU_SIZE],
                                    const Bool          checkDecorrelation,
                                    const ComponentID   compID,
                                          TComTU        &rTu,
                                          QpParam       &cQP
                                          DEBUG_STRING_FN_DECLARE(sDebug)
                                  );
  Void  xRecurIntraCodingQTTUCSC  ( TComYuv*       pcOrgYuv,
                                    TComYuv*       pcPredYuv,
                                    TComYuv*       pcResiYuv,
                                    Distortion&    PUDistY,
                                    Distortion&    PUDistC,
                                    Double&        dPUCost,
                                    TComTU&        rTu,
                                    Bool           bTestMaxTUSize,
                                    ACTRDTestTypes eACTRDTestType
                                    DEBUG_STRING_FN_DECLARE(sDebug)
                                   );

  Void  xRecurIntraCodingQTCSC    ( TComYuv*     pcOrgYuv,
                                    TComYuv*     pcPredYuv,
                                    TComYuv*     pcResiYuv,
                                    Distortion&  PUDistY,
                                    Distortion&  PUDistC,
                                    Double&      dPUCost,
                                    TComTU&      rTu,
                                    Bool         bTestMaxTUSize
                                    DEBUG_STRING_FN_DECLARE(sDebug)
                                  );

  Void xEstimateInterResidualQTTUCSC( TComYuv* pcResi, Double &rdCost, UInt &bits, Distortion &dist, TComTU &rTu, TComYuv* pcOrgResi, ACTRDTestTypes eACTRDtype DEBUG_STRING_FN_DECLARE(sDebug) );

  // Palette related functions
  Void  xInitTBCTable();
  UShort xGetTruncBinBits                  ( const UInt index, const UInt maxSymbolP1 );
  UShort xGetEscapeNumBins                 ( const Pel val );

  Void  xDerivePaletteLossy                ( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost );
  Void  xDerivePaletteLossyIterative       ( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost );
  UInt  xFindCandidatePalettePredictors    ( UInt paletteIndBest[], TComDataCU* pcCU, Pel *Palette[3], Pel* pPred[3], UInt paletteSizeTemp, UInt maxNoPredInd );
  Void  xDerivePaletteLossyForcePrediction ( TComDataCU *pcCU, Pel *Palette[3], Pel *pSrc[3], UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost );
  Void  xDerivePaletteLossless             ( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt &paletteSize );
  Bool  xCalLeftRun                        ( TComDataCU* pcCU, Pel* pValue, UChar * pSPoint, UInt startPos, UInt total, UInt &run, UChar* pEscapeFlag );
  Bool  xCalAboveRun                       ( TComDataCU* pcCU, Pel* pValue, UChar * pSPoint, UInt width, UInt startPos, UInt total, UInt &run, UChar* pEscapeFlag );
  Void  xCalcPixelPred                     ( TComDataCU* pcCU, Pel* pOrg [3], Pel*paPixelValue[3], Pel*paRecoValue[3], UInt width, UInt strideOrg, UInt startPos );
  Double xCalcPixelPredRD                  ( TComDataCU* pcCU, Pel pOrg[3], TComRdCost *pcCost, UInt *error, Bool discardChroma = false );
  UInt  xGetTruncatedBinBits               ( UInt symbol, UInt maxSymbol );
  UInt  xGetEpExGolombNumBins              ( UInt symbol, UInt count );
  Void  xPreCalcPaletteIndex               ( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt paletteSize );
  Void  xPreCalcPaletteIndexRD             ( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt calcErrorBits );
  Void  xReorderPalette                    ( TComDataCU* pcCU, Pel *Palette[3], UInt numComp );
  Void  xRotationScan                      ( Pel* pLevel, UInt width, UInt height );

  Void  xDeriveRun (TComDataCU* pcCU, Pel* pOrg [3],  Pel* pValue, UChar* pSPoint, Pel *pRecoValue[], Pel *pPixelRec[], TCoeff* pRun, UInt width, UInt height,  UInt strideOrg);
  Double xGetRunBits(TComDataCU* pcCU, Pel *pValue, UInt startPos, UInt run, PaletteRunMode paletteRunMode, UInt64 *allBits, UInt64 *indexBits, UInt64 *runBits);
  UInt preCalcRD(TComDataCU* pcCU, Pel *Palette[3], UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt iterNumber);
  Void preCalcRDMerge(TComDataCU* pcCU, UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt *errorOrig, UInt *errorNew, UInt calcErrBits);
  UInt calcPaletteIndexPredAndBits(Int iMaxSymbol, UInt idxStart, UInt width, UInt *predIndex, UInt *currIndex);
  UInt calcPaletteEndPosition(UInt copyPixels[], UInt positionInit, UInt run);
  UInt calcPaletteStartCopy(UInt positionInit, UInt positionCurrSegment, UInt width);
  UInt findPaletteSegment(PaletteInfoStruct *paletteElement, UInt idxStart, UInt indexMaxSize, UInt width, UInt total,
                          UInt copyPixels[], Int restrictLevelRun, UInt calcErrBits);
  UInt calcPaletteErrorCopy(UInt idxStart, UInt run, UInt width, UInt *mode);
  UInt64 calcPaletteErrorLevel(Int idxStart, UInt run, UInt paletteIdx);
  Void modifyPaletteSegment(UInt width, UInt idxStart, UInt paletteMode, UInt paletteIdx, UInt run);
};// END CLASS DEFINITION TEncSearch

//! \}

#endif // __TENCSEARCH__
