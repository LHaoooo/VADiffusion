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

/** \file     TEncSearch.cpp
 \brief    encoder search class
 */

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComRom.h"
#include "TLibCommon/TComMotionInfo.h"
#include "TEncSearch.h"
#include "TLibCommon/TComTU.h"
#include "TLibCommon/Debug.h"
#include <math.h>
#include <limits>


//! \ingroup TLibEncoder
//! \{

static const TComMv s_acMvRefineH[9] =
{
  TComMv(  0,  0 ), // 0
  TComMv(  0, -1 ), // 1
  TComMv(  0,  1 ), // 2
  TComMv( -1,  0 ), // 3
  TComMv(  1,  0 ), // 4
  TComMv( -1, -1 ), // 5
  TComMv(  1, -1 ), // 6
  TComMv( -1,  1 ), // 7
  TComMv(  1,  1 )  // 8
};

static const TComMv s_acMvRefineQ[9] =
{
  TComMv(  0,  0 ), // 0
  TComMv(  0, -1 ), // 1
  TComMv(  0,  1 ), // 2
  TComMv( -1, -1 ), // 5
  TComMv(  1, -1 ), // 6
  TComMv( -1,  0 ), // 3
  TComMv(  1,  0 ), // 4
  TComMv( -1,  1 ), // 7
  TComMv(  1,  1 )  // 8
};

static Void offsetSubTUCBFs(TComTU &rTu, const ComponentID compID)
{
        TComDataCU *pcCU              = rTu.getCU();
  const UInt        uiTrDepth         = rTu.GetTransformDepthRel();
  const UInt        uiAbsPartIdx      = rTu.GetAbsPartIdxTU(compID);
  const UInt        partIdxesPerSubTU = rTu.GetAbsPartIdxNumParts(compID) >> 1;

  //move the CBFs down a level and set the parent CBF

  UChar subTUCBF[2];
  UChar combinedSubTUCBF = 0;

  for (UInt subTU = 0; subTU < 2; subTU++)
  {
    const UInt subTUAbsPartIdx = uiAbsPartIdx + (subTU * partIdxesPerSubTU);

    subTUCBF[subTU]   = pcCU->getCbf(subTUAbsPartIdx, compID, uiTrDepth);
    combinedSubTUCBF |= subTUCBF[subTU];
  }

  for (UInt subTU = 0; subTU < 2; subTU++)
  {
    const UInt subTUAbsPartIdx = uiAbsPartIdx + (subTU * partIdxesPerSubTU);
    const UChar compositeCBF = (subTUCBF[subTU] << 1) | combinedSubTUCBF;

    pcCU->setCbfPartRange((compositeCBF << uiTrDepth), compID, subTUAbsPartIdx, partIdxesPerSubTU);
  }
}


TEncSearch::TEncSearch()
: m_puhQTTempTrIdx(NULL)
, m_pcQTTempTComYuv(NULL)
, m_pcEncCfg (NULL)
, m_pcTrQuant (NULL)
, m_pcRdCost (NULL)
, m_pcEntropyCoder (NULL)
, m_iSearchRange (0)
, m_bipredSearchRange (0)
, m_motionEstimationSearchMethod (MESEARCH_FULL)
, m_pppcRDSbacCoder (NULL)
, m_pcRDGoOnSbacCoder (NULL)
, m_pTempPel (NULL)
, m_isInitialized (false)
, m_paletteErrLimit(3)
, m_truncBinBits(NULL)
, m_escapeNumBins(NULL)
{
  for (UInt ch=0; ch<MAX_NUM_COMPONENT; ch++)
  {
    m_ppcQTTempCoeff[ch]                           = NULL;
#if ADAPTIVE_QP_SELECTION
    m_ppcQTTempArlCoeff[ch]                        = NULL;
#endif
    m_puhQTTempCbf[ch]                             = NULL;
    m_phQTTempCrossComponentPredictionAlpha[ch]    = NULL;
    m_pSharedPredTransformSkip[ch]                 = NULL;
    m_pcQTTempTUCoeff[ch]                          = NULL;
#if ADAPTIVE_QP_SELECTION
    m_ppcQTTempTUArlCoeff[ch]                      = NULL;
#endif
    m_puhQTTempTransformSkipFlag[ch]               = NULL;
    m_pcACTTempTUCoeff[ch]                         = NULL;
#if ADAPTIVE_QP_SELECTION
    m_ppcACTTempTUArlCoeff[ch]                     = NULL;
#endif
  }

  m_pcIntraBCHashTable = NULL;
  m_pcIntraBCHashTable = new IntraBCHashNode**[INTRABC_HASH_DEPTH];
  for(int i = 0; i < INTRABC_HASH_DEPTH; i++)
  {
    m_pcIntraBCHashTable[i] = new IntraBCHashNode*[INTRABC_HASH_TABLESIZE];
    memset(m_pcIntraBCHashTable[i], 0, INTRABC_HASH_TABLESIZE * sizeof(IntraBCHashNode*));
  }

  m_pcQTTempTComYuvCS                              = NULL;
  m_pcNoCorrYuvTmp                                 = NULL;
  m_puhQTTempACTFlag                               = NULL;

  m_paOriginalLevel  = (Pel*)xMalloc(Pel , MAX_CU_SIZE * MAX_CU_SIZE);

  UInt group=0, k=0, j, uiTotal=32*32;

  while (k<uiTotal)
  {
    if (group<2)
    {
      m_runGolombGroups[k]=group+1;
      k++; group++;
      if (k>=uiTotal)
      {
        break;
      }
    }
    else
    {
      for (j=0; j<(1<<(group-1)); j++)
      {
        m_runGolombGroups[k+j]=2*group;
      }
      k+=(1<<(group-1));
      group++;
    }
    if (k>=uiTotal)
    {
      break;
    }
  }

  for(Int i=0; i<MAX_NUM_COMPONENT; i++)
  {
    m_paBestLevel[i]  = (Pel*)xMalloc(Pel , MAX_CU_SIZE * MAX_CU_SIZE);
  }
  m_paBestSPoint = (UChar*)xMalloc(UChar , MAX_CU_SIZE * MAX_CU_SIZE);
  m_paBestRun    = (TCoeff*)xMalloc(TCoeff , MAX_CU_SIZE * MAX_CU_SIZE);
  m_paBestEscapeFlag = (UChar*)xMalloc(UChar , MAX_CU_SIZE * MAX_CU_SIZE);

  for(Int i=0; i<MAX_NUM_COMPONENT; i++)
  {
    m_paLevelStoreRD[i]  = (Pel*)xMalloc(Pel , MAX_CU_SIZE * MAX_CU_SIZE);
  }
  m_paSPointStoreRD = (UChar*)xMalloc(UChar , MAX_CU_SIZE * MAX_CU_SIZE);
  m_paRunStoreRD    = (TCoeff*)xMalloc(TCoeff , MAX_CU_SIZE * MAX_CU_SIZE);
  m_paEscapeFlagStoreRD = (UChar*)xMalloc(UChar , MAX_CU_SIZE * MAX_CU_SIZE);

  for (Int i=0; i<MAX_NUM_REF_LIST_ADAPT_SR; i++)
  {
    memset (m_aaiAdaptSR[i], 0, MAX_IDX_ADAPT_SR * sizeof (Int));
  }
  for (Int i=0; i<AMVP_MAX_NUM_CANDS+1; i++)
  {
    memset (m_auiMVPIdxCost[i], 0, (AMVP_MAX_NUM_CANDS+1) * sizeof (UInt) );
  }

  setWpScalingDistParam( NULL, -1, REF_PIC_LIST_X );
}


Void TEncSearch::destroy()
{
  assert (m_isInitialized);
  if ( m_pTempPel )
  {
    delete [] m_pTempPel;
    m_pTempPel = NULL;
  }

  if ( m_pcEncCfg )
  {
    const UInt uiNumLayersAllocated = m_pcEncCfg->getQuadtreeTULog2MaxSize()-m_pcEncCfg->getQuadtreeTULog2MinSize()+1;

    for (UInt ch=0; ch<MAX_NUM_COMPONENT; ch++)
    {
      for (UInt layer = 0; layer < uiNumLayersAllocated; layer++)
      {
        delete[] m_ppcQTTempCoeff[ch][layer];
#if ADAPTIVE_QP_SELECTION
        delete[] m_ppcQTTempArlCoeff[ch][layer];
#endif
      }
      delete[] m_ppcQTTempCoeff[ch];
      delete[] m_puhQTTempCbf[ch];
#if ADAPTIVE_QP_SELECTION
      delete[] m_ppcQTTempArlCoeff[ch];
#endif
    }

    for( UInt layer = 0; layer < uiNumLayersAllocated; layer++ )
    {
      m_pcQTTempTComYuv[layer].destroy();
      m_pcQTTempTComYuvCS[layer].destroy();
      m_pcNoCorrYuvTmp[layer].destroy();
    }
  }

  delete[] m_puhQTTempTrIdx;
  delete[] m_pcQTTempTComYuv;
  delete[] m_puhQTTempACTFlag;
  delete[] m_pcQTTempTComYuvCS;
  delete[] m_pcNoCorrYuvTmp;

  for (UInt ch=0; ch<MAX_NUM_COMPONENT; ch++)
  {
    delete[] m_pSharedPredTransformSkip[ch];
    delete[] m_pcQTTempTUCoeff[ch];
#if ADAPTIVE_QP_SELECTION
    delete[] m_ppcQTTempTUArlCoeff[ch];
#endif
    delete[] m_phQTTempCrossComponentPredictionAlpha[ch];
    delete[] m_puhQTTempTransformSkipFlag[ch];
    delete[] m_pcACTTempTUCoeff[ch];
#if ADAPTIVE_QP_SELECTION
    delete[] m_ppcACTTempTUArlCoeff[ch];
#endif
  }
  m_pcQTTempTransformSkipTComYuv.destroy();
  m_pcACTTempTransformSkipTComYuv.destroy();

  if(m_pcIntraBCHashTable)
  {
    for(int iDepth = 0; iDepth < INTRABC_HASH_DEPTH; iDepth++)
    {
      if(m_pcIntraBCHashTable[iDepth])
      {
        delete[]  m_pcIntraBCHashTable[iDepth];
      }
    }
    delete[] m_pcIntraBCHashTable;
  }

  m_pcIntraBCHashTable = NULL;


  m_tmpYuvPred.destroy();
  if (m_paOriginalLevel)  { xFree(m_paOriginalLevel);  m_paOriginalLevel=NULL;  }
  for(Int i=0; i<MAX_NUM_COMPONENT; i++)
  {
    if (m_paBestLevel[i])      { xFree(m_paBestLevel[i]);  m_paBestLevel[i]=NULL;  }
  }
  if (m_paBestSPoint)     { xFree(m_paBestSPoint); m_paBestSPoint=NULL; }
  if (m_paBestRun)        { xFree(m_paBestRun);    m_paBestRun=NULL;    }
  if (m_paBestEscapeFlag) { xFree(m_paBestEscapeFlag);    m_paBestEscapeFlag=NULL;    }

  for(Int i=0; i<MAX_NUM_COMPONENT; i++)
  {
    if (m_paLevelStoreRD[i])      { xFree(m_paLevelStoreRD[i]);  m_paLevelStoreRD[i]=NULL;  }
  }
  if (m_paSPointStoreRD)     { xFree(m_paSPointStoreRD); m_paSPointStoreRD=NULL; }
  if (m_paRunStoreRD)        { xFree(m_paRunStoreRD);    m_paRunStoreRD=NULL;    }
  if (m_paEscapeFlagStoreRD) { xFree(m_paEscapeFlagStoreRD);    m_paEscapeFlagStoreRD=NULL;    }

  m_isInitialized = false;

  if( m_truncBinBits )
  {
    for( UInt i = 0; i < m_maxSymbolSize; i++ )
    {
      if( m_truncBinBits[i] )
      {
        delete[] m_truncBinBits[i];
        m_truncBinBits[i] = NULL;
      }
    }

    delete[] m_truncBinBits;
    m_truncBinBits = NULL;
  }

  if( m_escapeNumBins )
  {
    delete[] m_escapeNumBins;
    m_escapeNumBins = NULL;
  }
}

TEncSearch::~TEncSearch()
{
  if (m_isInitialized)
  {
    destroy();
  }
}




Void TEncSearch::init(TEncCfg*       pcEncCfg,
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
                      TEncSbac*      pcRDGoOnSbacCoder
                      )
{
  assert (!m_isInitialized);
  m_pcEncCfg                     = pcEncCfg;
  m_pcTrQuant                    = pcTrQuant;
  m_iSearchRange                 = iSearchRange;
  m_bipredSearchRange            = bipredSearchRange;
  m_motionEstimationSearchMethod = motionEstimationSearchMethod;
  m_pcEntropyCoder               = pcEntropyCoder;
  m_pcRdCost                     = pcRdCost;

  m_pppcRDSbacCoder              = pppcRDSbacCoder;
  m_pcRDGoOnSbacCoder            = pcRDGoOnSbacCoder;

  m_numBVs = 0;

  for (UInt iDir = 0; iDir < MAX_NUM_REF_LIST_ADAPT_SR; iDir++)
  {
    for (UInt iRefIdx = 0; iRefIdx < MAX_IDX_ADAPT_SR; iRefIdx++)
    {
      m_aaiAdaptSR[iDir][iRefIdx] = iSearchRange;
    }
  }

  // initialize motion cost
  for( Int iNum = 0; iNum < AMVP_MAX_NUM_CANDS+1; iNum++)
  {
    for( Int iIdx = 0; iIdx < AMVP_MAX_NUM_CANDS; iIdx++)
    {
      if (iIdx < iNum)
      {
        m_auiMVPIdxCost[iIdx][iNum] = xGetMvpIdxBits(iIdx, iNum);
      }
      else
      {
        m_auiMVPIdxCost[iIdx][iNum] = MAX_INT;
      }
    }
  }

  const ChromaFormat cform=pcEncCfg->getChromaFormatIdc();
  initTempBuff(cform);
  if( pcEncCfg->getUsePaletteMode() )
  {
    xInitTBCTable();
  }
  m_pTempPel = new Pel[maxCUWidth*maxCUHeight];

  const UInt uiNumLayersToAllocate = pcEncCfg->getQuadtreeTULog2MaxSize()-pcEncCfg->getQuadtreeTULog2MinSize()+1;
  const UInt uiNumPartitions = 1<<(maxTotalCUDepth<<1);
  for (UInt ch=0; ch<MAX_NUM_COMPONENT; ch++)
  {
    const UInt csx=::getComponentScaleX(ComponentID(ch), cform);
    const UInt csy=::getComponentScaleY(ComponentID(ch), cform);
    m_ppcQTTempCoeff[ch] = new TCoeff* [uiNumLayersToAllocate];
#if ADAPTIVE_QP_SELECTION
    m_ppcQTTempArlCoeff[ch]  = new TCoeff*[uiNumLayersToAllocate];
#endif
    m_puhQTTempCbf[ch] = new UChar  [uiNumPartitions];

    for (UInt layer = 0; layer < uiNumLayersToAllocate; layer++)
    {
      m_ppcQTTempCoeff[ch][layer] = new TCoeff[(maxCUWidth*maxCUHeight)>>(csx+csy)];
#if ADAPTIVE_QP_SELECTION
      m_ppcQTTempArlCoeff[ch][layer]  = new TCoeff[(maxCUWidth*maxCUHeight)>>(csx+csy) ];
#endif
    }

    m_phQTTempCrossComponentPredictionAlpha[ch]    = new SChar  [uiNumPartitions];
    m_pSharedPredTransformSkip[ch]                 = new Pel   [MAX_CU_SIZE*MAX_CU_SIZE];
    m_pcQTTempTUCoeff[ch]                          = new TCoeff[MAX_CU_SIZE*MAX_CU_SIZE];
#if ADAPTIVE_QP_SELECTION
    m_ppcQTTempTUArlCoeff[ch]                      = new TCoeff[MAX_CU_SIZE*MAX_CU_SIZE];
#endif
    m_puhQTTempTransformSkipFlag[ch]               = new UChar [uiNumPartitions];
    m_pcACTTempTUCoeff[ch]                         = new TCoeff[MAX_CU_SIZE*MAX_CU_SIZE];
#if ADAPTIVE_QP_SELECTION
    m_ppcACTTempTUArlCoeff[ch]                     = new TCoeff[MAX_CU_SIZE*MAX_CU_SIZE];
#endif
  }
  m_puhQTTempTrIdx   = new UChar  [uiNumPartitions];
  m_pcQTTempTComYuv  = new TComYuv[uiNumLayersToAllocate];
  m_puhQTTempACTFlag   = new Bool   [uiNumPartitions];
  m_pcQTTempTComYuvCS  = new TComYuv[uiNumLayersToAllocate];
  m_pcNoCorrYuvTmp     = new TComYuv[uiNumLayersToAllocate];
  for( UInt ui = 0; ui < uiNumLayersToAllocate; ++ui )
  {
    m_pcQTTempTComYuv[ui].create( maxCUWidth, maxCUHeight, pcEncCfg->getChromaFormatIdc() );
    m_pcQTTempTComYuvCS[ui].create( maxCUWidth, maxCUHeight, pcEncCfg->getChromaFormatIdc() );
    m_pcNoCorrYuvTmp[ui].create( (maxCUWidth>>ui), (maxCUHeight>>ui), pcEncCfg->getChromaFormatIdc() );
  }
  m_pcQTTempTransformSkipTComYuv.create( maxCUWidth, maxCUHeight, pcEncCfg->getChromaFormatIdc() );
  m_pcACTTempTransformSkipTComYuv.create( maxCUWidth, maxCUHeight, pcEncCfg->getChromaFormatIdc() );
  m_tmpYuvPred.create(MAX_CU_SIZE, MAX_CU_SIZE, pcEncCfg->getChromaFormatIdc());
  m_isInitialized = true;
}


__inline Void TEncSearch::xTZSearchHelp( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const Int iSearchX, const Int iSearchY, const UChar ucPointNr, const UInt uiDistance )
{
  Distortion  uiSad = 0;

  const Pel* const  piRefSrch = rcStruct.piRefY + iSearchY * rcStruct.iYStride + iSearchX;

  //-- jclee for using the SAD function pointer
  m_pcRdCost->setDistParam( pcPatternKey, piRefSrch, rcStruct.iYStride,  m_cDistParam );

  setDistParamComp(COMPONENT_Y);

  // distortion
  m_cDistParam.bitDepth = pcPatternKey->getBitDepthY();
  m_cDistParam.m_maximumDistortionForEarlyExit = rcStruct.uiBestSad;

  if((m_pcEncCfg->getRestrictMESampling() == false) && m_pcEncCfg->getMotionEstimationSearchMethod() == MESEARCH_SELECTIVE)
  {
    Int isubShift = 0;
    // motion cost
    Distortion uiBitCost = m_pcRdCost->getCostOfVectorWithPredictor( iSearchX, iSearchY );

    // Skip search if bit cost is already larger than best SAD
    if (uiBitCost < rcStruct.uiBestSad)
    {
      if ( m_cDistParam.iRows > 32 )
      {
        m_cDistParam.iSubShift = 4;
      }
      else if ( m_cDistParam.iRows > 16 )
      {
        m_cDistParam.iSubShift = 3;
      }
      else if ( m_cDistParam.iRows > 8 )
      {
        m_cDistParam.iSubShift = 2;
      }
      else
      {
        m_cDistParam.iSubShift = 1;
      }

      Distortion uiTempSad = m_cDistParam.DistFunc( &m_cDistParam );
      if((uiTempSad + uiBitCost) < rcStruct.uiBestSad)
      {
        uiSad += uiTempSad >>  m_cDistParam.iSubShift;
        while(m_cDistParam.iSubShift > 0)
        {
          isubShift         = m_cDistParam.iSubShift -1;
          m_cDistParam.pOrg = pcPatternKey->getROIY() + (pcPatternKey->getPatternLStride() << isubShift);
          m_cDistParam.pCur = piRefSrch + (rcStruct.iYStride << isubShift);
          uiTempSad = m_cDistParam.DistFunc( &m_cDistParam );
          uiSad += uiTempSad >>  m_cDistParam.iSubShift;
          if(((uiSad << isubShift) + uiBitCost) > rcStruct.uiBestSad)
          {
            break;
          }

          m_cDistParam.iSubShift--;
        }

        if(m_cDistParam.iSubShift == 0)
        {
          uiSad += uiBitCost;
          if( uiSad < rcStruct.uiBestSad )
          {
            rcStruct.uiBestSad      = uiSad;
            rcStruct.iBestX         = iSearchX;
            rcStruct.iBestY         = iSearchY;
            rcStruct.uiBestDistance = uiDistance;
            rcStruct.uiBestRound    = 0;
            rcStruct.ucPointNr      = ucPointNr;
            m_cDistParam.m_maximumDistortionForEarlyExit = uiSad;
          }
        }
      }
    }
  }
  else
  {
    // fast encoder decision: use subsampled SAD when rows > 8 for integer ME
    if ( m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE1 || m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE3 )
    {
      if ( m_cDistParam.iRows > 8 )
      {
        m_cDistParam.iSubShift = 1;
      }
    }

    uiSad = m_cDistParam.DistFunc( &m_cDistParam );

    // only add motion cost if uiSad is smaller than best. Otherwise pointless
    // to add motion cost.
    if( uiSad < rcStruct.uiBestSad )
    {
      // motion cost
      uiSad += m_pcRdCost->getCostOfVectorWithPredictor( iSearchX, iSearchY );

      if( uiSad < rcStruct.uiBestSad )
      {
        rcStruct.uiBestSad      = uiSad;
        rcStruct.iBestX         = iSearchX;
        rcStruct.iBestY         = iSearchY;
        rcStruct.uiBestDistance = uiDistance;
        rcStruct.uiBestRound    = 0;
        rcStruct.ucPointNr      = ucPointNr;
        m_cDistParam.m_maximumDistortionForEarlyExit = uiSad;
      }
    }
  }
}

__inline Void TEncSearch::xTZ2PointSearch( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const TComMv* const pcMvSrchRngLT, const TComMv* const pcMvSrchRngRB )
{
  Int   iSrchRngHorLeft   = pcMvSrchRngLT->getHor();
  Int   iSrchRngHorRight  = pcMvSrchRngRB->getHor();
  Int   iSrchRngVerTop    = pcMvSrchRngLT->getVer();
  Int   iSrchRngVerBottom = pcMvSrchRngRB->getVer();

  // 2 point search,                   //   1 2 3
  // check only the 2 untested points  //   4 0 5
  // around the start point            //   6 7 8
  Int iStartX = rcStruct.iBestX;
  Int iStartY = rcStruct.iBestY;
  switch( rcStruct.ucPointNr )
  {
    case 1:
    {
      if ( (iStartX - 1) >= iSrchRngHorLeft )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY, 0, 2 );
      }
      if ( (iStartY - 1) >= iSrchRngVerTop )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iStartY - 1, 0, 2 );
      }
    }
      break;
    case 2:
    {
      if ( (iStartY - 1) >= iSrchRngVerTop )
      {
        if ( (iStartX - 1) >= iSrchRngHorLeft )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY - 1, 0, 2 );
        }
        if ( (iStartX + 1) <= iSrchRngHorRight )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY - 1, 0, 2 );
        }
      }
    }
      break;
    case 3:
    {
      if ( (iStartY - 1) >= iSrchRngVerTop )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iStartY - 1, 0, 2 );
      }
      if ( (iStartX + 1) <= iSrchRngHorRight )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY, 0, 2 );
      }
    }
      break;
    case 4:
    {
      if ( (iStartX - 1) >= iSrchRngHorLeft )
      {
        if ( (iStartY + 1) <= iSrchRngVerBottom )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY + 1, 0, 2 );
        }
        if ( (iStartY - 1) >= iSrchRngVerTop )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY - 1, 0, 2 );
        }
      }
    }
      break;
    case 5:
    {
      if ( (iStartX + 1) <= iSrchRngHorRight )
      {
        if ( (iStartY - 1) >= iSrchRngVerTop )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY - 1, 0, 2 );
        }
        if ( (iStartY + 1) <= iSrchRngVerBottom )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY + 1, 0, 2 );
        }
      }
    }
      break;
    case 6:
    {
      if ( (iStartX - 1) >= iSrchRngHorLeft )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY , 0, 2 );
      }
      if ( (iStartY + 1) <= iSrchRngVerBottom )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iStartY + 1, 0, 2 );
      }
    }
      break;
    case 7:
    {
      if ( (iStartY + 1) <= iSrchRngVerBottom )
      {
        if ( (iStartX - 1) >= iSrchRngHorLeft )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX - 1, iStartY + 1, 0, 2 );
        }
        if ( (iStartX + 1) <= iSrchRngHorRight )
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY + 1, 0, 2 );
        }
      }
    }
      break;
    case 8:
    {
      if ( (iStartX + 1) <= iSrchRngHorRight )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX + 1, iStartY, 0, 2 );
      }
      if ( (iStartY + 1) <= iSrchRngVerBottom )
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iStartY + 1, 0, 2 );
      }
    }
      break;
    default:
    {
      assert( false );
    }
      break;
  } // switch( rcStruct.ucPointNr )
}




__inline Void TEncSearch::xTZ8PointSquareSearch( const TComPattern* const pcPatternKey, IntTZSearchStruct& rcStruct, const TComMv* const pcMvSrchRngLT, const TComMv* const pcMvSrchRngRB, const Int iStartX, const Int iStartY, const Int iDist )
{
  const Int   iSrchRngHorLeft   = pcMvSrchRngLT->getHor();
  const Int   iSrchRngHorRight  = pcMvSrchRngRB->getHor();
  const Int   iSrchRngVerTop    = pcMvSrchRngLT->getVer();
  const Int   iSrchRngVerBottom = pcMvSrchRngRB->getVer();

  // 8 point search,                   //   1 2 3
  // search around the start point     //   4 0 5
  // with the required  distance       //   6 7 8
  assert( iDist != 0 );
  const Int iTop        = iStartY - iDist;
  const Int iBottom     = iStartY + iDist;
  const Int iLeft       = iStartX - iDist;
  const Int iRight      = iStartX + iDist;
  rcStruct.uiBestRound += 1;

  if ( iTop >= iSrchRngVerTop ) // check top
  {
    if ( iLeft >= iSrchRngHorLeft ) // check top left
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iTop, 1, iDist );
    }
    // top middle
    xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop, 2, iDist );

    if ( iRight <= iSrchRngHorRight ) // check top right
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iRight, iTop, 3, iDist );
    }
  } // check top
  if ( iLeft >= iSrchRngHorLeft ) // check middle left
  {
    xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iStartY, 4, iDist );
  }
  if ( iRight <= iSrchRngHorRight ) // check middle right
  {
    xTZSearchHelp( pcPatternKey, rcStruct, iRight, iStartY, 5, iDist );
  }
  if ( iBottom <= iSrchRngVerBottom ) // check bottom
  {
    if ( iLeft >= iSrchRngHorLeft ) // check bottom left
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iBottom, 6, iDist );
    }
    // check bottom middle
    xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 7, iDist );

    if ( iRight <= iSrchRngHorRight ) // check bottom right
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iRight, iBottom, 8, iDist );
    }
  } // check bottom
}




__inline Void TEncSearch::xTZ8PointDiamondSearch( const TComPattern*const  pcPatternKey,
                                                  IntTZSearchStruct& rcStruct,
                                                  const TComMv*const  pcMvSrchRngLT,
                                                  const TComMv*const  pcMvSrchRngRB,
                                                  const Int iStartX,
                                                  const Int iStartY,
                                                  const Int iDist,
                                                  const Bool bCheckCornersAtDist1,
                                                  Bool bSkipLeftDist2,
                                                  Bool bSkipTopDist2 )
{
  const Int   iSrchRngHorLeft   = pcMvSrchRngLT->getHor();
  const Int   iSrchRngHorRight  = pcMvSrchRngRB->getHor();
  const Int   iSrchRngVerTop    = pcMvSrchRngLT->getVer();
  const Int   iSrchRngVerBottom = pcMvSrchRngRB->getVer();

  // 8 point search,                   //   1 2 3
  // search around the start point     //   4 0 5
  // with the required  distance       //   6 7 8
  assert ( iDist != 0 );
  const Int iTop        = iStartY - iDist;
  const Int iBottom     = iStartY + iDist;
  const Int iLeft       = iStartX - iDist;
  const Int iRight      = iStartX + iDist;
  rcStruct.uiBestRound += 1;

  if (bSkipLeftDist2 || bSkipTopDist2)
  {
    assert(iDist == 2);
  }

  if ( iDist == 1 )
  {
    if ( iTop >= iSrchRngVerTop ) // check top
    {
      if (bCheckCornersAtDist1)
      {
        if ( iLeft >= iSrchRngHorLeft) // check top-left
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iTop, 1, iDist );
        }
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop, 2, iDist );
        if ( iRight <= iSrchRngHorRight ) // check middle right
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iRight, iTop, 3, iDist );
        }
      }
      else
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop, 2, iDist );
      }
    }
    if ( iLeft >= iSrchRngHorLeft ) // check middle left
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iStartY, 4, iDist );
    }
    if ( iRight <= iSrchRngHorRight ) // check middle right
    {
      xTZSearchHelp( pcPatternKey, rcStruct, iRight, iStartY, 5, iDist );
    }
    if ( iBottom <= iSrchRngVerBottom ) // check bottom
    {
      if (bCheckCornersAtDist1)
      {
        if ( iLeft >= iSrchRngHorLeft) // check top-left
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iBottom, 6, iDist );
        }
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 7, iDist );
        if ( iRight <= iSrchRngHorRight ) // check middle right
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iRight, iBottom, 8, iDist );
        }
      }
      else
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 7, iDist );
      }
    }
  }
  else
  {
    if ( iDist <= 8 )
    {
      const Int iTop_2      = iStartY - (iDist>>1);
      const Int iBottom_2   = iStartY + (iDist>>1);
      const Int iLeft_2     = iStartX - (iDist>>1);
      const Int iRight_2    = iStartX + (iDist>>1);

      if (  iTop >= iSrchRngVerTop && iLeft >= iSrchRngHorLeft &&
          iRight <= iSrchRngHorRight && iBottom <= iSrchRngVerBottom ) // check border
      {
        if (!bSkipTopDist2)
        {
          xTZSearchHelp(pcPatternKey, rcStruct, iStartX, iTop,      2, iDist    );
        }
        xTZSearchHelp( pcPatternKey, rcStruct, iLeft_2,  iTop_2,    1, iDist>>1 );
        xTZSearchHelp( pcPatternKey, rcStruct, iRight_2, iTop_2,    3, iDist>>1 );
        if (!bSkipLeftDist2)
        {
          xTZSearchHelp(pcPatternKey, rcStruct, iLeft, iStartY,     4, iDist    );
        }
        xTZSearchHelp( pcPatternKey, rcStruct, iRight,   iStartY,   5, iDist    );
        xTZSearchHelp( pcPatternKey, rcStruct, iLeft_2,  iBottom_2, 6, iDist>>1 );
        xTZSearchHelp( pcPatternKey, rcStruct, iRight_2, iBottom_2, 8, iDist>>1 );
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX,  iBottom,   7, iDist    );
      }
      else // check border
      {
        if ( iTop >= iSrchRngVerTop && !bSkipTopDist2 ) // check top
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop, 2, iDist );
        }
        if ( iTop_2 >= iSrchRngVerTop ) // check half top
        {
          if ( iLeft_2 >= iSrchRngHorLeft ) // check half left
          {
            xTZSearchHelp( pcPatternKey, rcStruct, iLeft_2, iTop_2, 1, (iDist>>1) );
          }
          if ( iRight_2 <= iSrchRngHorRight ) // check half right
          {
            xTZSearchHelp( pcPatternKey, rcStruct, iRight_2, iTop_2, 3, (iDist>>1) );
          }
        } // check half top
        if ( iLeft >= iSrchRngHorLeft && !bSkipLeftDist2 ) // check left
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iStartY, 4, iDist );
        }
        if ( iRight <= iSrchRngHorRight ) // check right
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iRight, iStartY, 5, iDist );
        }
        if ( iBottom_2 <= iSrchRngVerBottom ) // check half bottom
        {
          if ( iLeft_2 >= iSrchRngHorLeft ) // check half left
          {
            xTZSearchHelp( pcPatternKey, rcStruct, iLeft_2, iBottom_2, 6, (iDist>>1) );
          }
          if ( iRight_2 <= iSrchRngHorRight ) // check half right
          {
            xTZSearchHelp( pcPatternKey, rcStruct, iRight_2, iBottom_2, 8, (iDist>>1) );
          }
        } // check half bottom
        if ( iBottom <= iSrchRngVerBottom ) // check bottom
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 7, iDist );
        }
      } // check border
    }
    else // iDist > 8
    {
      if ( iTop >= iSrchRngVerTop && iLeft >= iSrchRngHorLeft &&
          iRight <= iSrchRngHorRight && iBottom <= iSrchRngVerBottom ) // check border
      {
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop,    0, iDist );
        xTZSearchHelp( pcPatternKey, rcStruct, iLeft,   iStartY, 0, iDist );
        xTZSearchHelp( pcPatternKey, rcStruct, iRight,  iStartY, 0, iDist );
        xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 0, iDist );
        for ( Int index = 1; index < 4; index++ )
        {
          const Int iPosYT = iTop    + ((iDist>>2) * index);
          const Int iPosYB = iBottom - ((iDist>>2) * index);
          const Int iPosXL = iStartX - ((iDist>>2) * index);
          const Int iPosXR = iStartX + ((iDist>>2) * index);
          xTZSearchHelp( pcPatternKey, rcStruct, iPosXL, iPosYT, 0, iDist );
          xTZSearchHelp( pcPatternKey, rcStruct, iPosXR, iPosYT, 0, iDist );
          xTZSearchHelp( pcPatternKey, rcStruct, iPosXL, iPosYB, 0, iDist );
          xTZSearchHelp( pcPatternKey, rcStruct, iPosXR, iPosYB, 0, iDist );
        }
      }
      else // check border
      {
        if ( iTop >= iSrchRngVerTop ) // check top
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iTop, 0, iDist );
        }
        if ( iLeft >= iSrchRngHorLeft ) // check left
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iLeft, iStartY, 0, iDist );
        }
        if ( iRight <= iSrchRngHorRight ) // check right
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iRight, iStartY, 0, iDist );
        }
        if ( iBottom <= iSrchRngVerBottom ) // check bottom
        {
          xTZSearchHelp( pcPatternKey, rcStruct, iStartX, iBottom, 0, iDist );
        }
        for ( Int index = 1; index < 4; index++ )
        {
          const Int iPosYT = iTop    + ((iDist>>2) * index);
          const Int iPosYB = iBottom - ((iDist>>2) * index);
          const Int iPosXL = iStartX - ((iDist>>2) * index);
          const Int iPosXR = iStartX + ((iDist>>2) * index);

          if ( iPosYT >= iSrchRngVerTop ) // check top
          {
            if ( iPosXL >= iSrchRngHorLeft ) // check left
            {
              xTZSearchHelp( pcPatternKey, rcStruct, iPosXL, iPosYT, 0, iDist );
            }
            if ( iPosXR <= iSrchRngHorRight ) // check right
            {
              xTZSearchHelp( pcPatternKey, rcStruct, iPosXR, iPosYT, 0, iDist );
            }
          } // check top
          if ( iPosYB <= iSrchRngVerBottom ) // check bottom
          {
            if ( iPosXL >= iSrchRngHorLeft ) // check left
            {
              xTZSearchHelp( pcPatternKey, rcStruct, iPosXL, iPosYB, 0, iDist );
            }
            if ( iPosXR <= iSrchRngHorRight ) // check right
            {
              xTZSearchHelp( pcPatternKey, rcStruct, iPosXR, iPosYB, 0, iDist );
            }
          } // check bottom
        } // for ...
      } // check border
    } // iDist <= 8
  } // iDist == 1
}

Distortion TEncSearch::xPatternRefinement( TComPattern* pcPatternKey,
                                           TComMv baseRefMv,
                                           Int iFrac, TComMv& rcMvFrac,
                                           Bool bAllowUseOfHadamard
                                         )
{
  Distortion  uiDist;
  Distortion  uiDistBest  = std::numeric_limits<Distortion>::max();
  UInt        uiDirecBest = 0;

  Pel*  piRefPos;
  Int iRefStride = m_filteredBlock[0][0].getStride(COMPONENT_Y);

  m_pcRdCost->setDistParam( pcPatternKey, m_filteredBlock[0][0].getAddr(COMPONENT_Y), iRefStride, 1, m_cDistParam, m_pcEncCfg->getUseHADME() && bAllowUseOfHadamard );

  const TComMv* pcMvRefine = (iFrac == 2 ? s_acMvRefineH : s_acMvRefineQ);

#if MCTS_ENC_CHECK
  UInt maxRefinements = 9;
  Int mvShift = 2;

  // filter length of sub-sample generation filter to be considered
  const UInt LumaLTSampleOffset = 3;
  const UInt LumaRBSampleOffset = 4;

  if (m_pcEncCfg->getTMCTSSEITileConstraint())
  {
    // if close to tile borders
    if ( pcPatternKey->getROIYPosX() + (baseRefMv.getHor() >> mvShift ) < pcPatternKey->getTileLeftTopPelPosX() +  LumaLTSampleOffset ||
         pcPatternKey->getROIYPosY() + (baseRefMv.getVer() >> mvShift ) < pcPatternKey->getTileLeftTopPelPosY() +  LumaLTSampleOffset ||
         pcPatternKey->getROIYPosX() + (baseRefMv.getHor() >> mvShift)  > pcPatternKey->getTileRightBottomPelPosX() - pcPatternKey->getROIYWidth() - LumaRBSampleOffset ||
         pcPatternKey->getROIYPosY() + (baseRefMv.getVer() >> mvShift)  > pcPatternKey->getTileRightBottomPelPosY() - pcPatternKey->getROIYHeight() - LumaRBSampleOffset  
       )
    {
      // only allow full pel positions to avoid filter dependency
      maxRefinements = 1;
    }
  }  

  for (UInt i = 0; i < maxRefinements; i++)
#else
  for (UInt i = 0; i < 9; i++)
#endif
  {
    if ( m_bSkipFracME && i > 0 )
    {
      break;
    }
    TComMv cMvTest = pcMvRefine[i];
    cMvTest += baseRefMv;

    Int horVal = cMvTest.getHor() * iFrac;
    Int verVal = cMvTest.getVer() * iFrac;
    piRefPos = m_filteredBlock[ verVal & 3 ][ horVal & 3 ].getAddr(COMPONENT_Y);
    if ( horVal == 2 && ( verVal & 1 ) == 0 )
    {
      piRefPos += 1;
    }
    if ( ( horVal & 1 ) == 0 && verVal == 2 )
    {
      piRefPos += iRefStride;
    }
    cMvTest = pcMvRefine[i];
    cMvTest += rcMvFrac;

    setDistParamComp(COMPONENT_Y);

    m_cDistParam.pCur = piRefPos;
    m_cDistParam.bitDepth = pcPatternKey->getBitDepthY();
    uiDist = m_cDistParam.DistFunc( &m_cDistParam );
    uiDist += m_pcRdCost->getCostOfVectorWithPredictor( cMvTest.getHor(), cMvTest.getVer() );

    if ( uiDist < uiDistBest )
    {
      uiDistBest  = uiDist;
      uiDirecBest = i;
      m_cDistParam.m_maximumDistortionForEarlyExit = uiDist;
    }
  }

  rcMvFrac = pcMvRefine[uiDirecBest];

  return uiDistBest;
}



Void
TEncSearch::xEncSubdivCbfQT(TComTU      &rTu,
                            Bool         bLuma,
                            Bool         bChroma )
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt uiAbsPartIdx         = rTu.GetAbsPartIdxTU();
  const UInt uiTrDepth            = rTu.GetTransformDepthRel();
  const UInt uiTrMode             = pcCU->getTransformIdx( uiAbsPartIdx );
  const UInt uiSubdiv             = ( uiTrMode > uiTrDepth ? 1 : 0 );
  const UInt uiLog2LumaTrafoSize  = rTu.GetLog2LumaTrSize();

  if( pcCU->isIntra(0) && pcCU->getPartitionSize(0) == SIZE_NxN && uiTrDepth == 0 )
  {
    assert( uiSubdiv );
  }
  else if( uiLog2LumaTrafoSize > pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() )
  {
    assert( uiSubdiv );
  }
  else if( uiLog2LumaTrafoSize == pcCU->getSlice()->getSPS()->getQuadtreeTULog2MinSize() )
  {
    assert( !uiSubdiv );
  }
  else if( uiLog2LumaTrafoSize == pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) )
  {
    assert( !uiSubdiv );
  }
  else
  {
    assert( uiLog2LumaTrafoSize > pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) );
    if( bLuma )
    {
      m_pcEntropyCoder->encodeTransformSubdivFlag( uiSubdiv, 5 - uiLog2LumaTrafoSize );
    }
  }

  if ( bChroma )
  {
    const UInt numberValidComponents = getNumberValidComponents(rTu.GetChromaFormat());
    for (UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
    {
      const ComponentID compID=ComponentID(ch);
      if( rTu.ProcessingAllQuadrants(compID) && (uiTrDepth==0 || pcCU->getCbf( uiAbsPartIdx, compID, uiTrDepth-1 ) ))
      {
        m_pcEntropyCoder->encodeQtCbf(rTu, compID, (uiSubdiv == 0));
      }
    }
  }

  if( uiSubdiv )
  {
    TComTURecurse tuRecurse(rTu, false);
    do
    {
      xEncSubdivCbfQT( tuRecurse, bLuma, bChroma );
    } while (tuRecurse.nextSection(rTu));
  }
  else
  {
    //===== Cbfs =====
    if( bLuma )
    {
      m_pcEntropyCoder->encodeQtCbf( rTu, COMPONENT_Y, true );
    }
  }
}




Void
TEncSearch::xEncCoeffQT(TComTU &rTu,
                        const ComponentID  component,
                        Bool         bRealCoeff )
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiTrDepth=rTu.GetTransformDepthRel();

  const UInt  uiTrMode        = pcCU->getTransformIdx( uiAbsPartIdx );
  const UInt  uiSubdiv        = ( uiTrMode > uiTrDepth ? 1 : 0 );

  if( uiSubdiv )
  {
    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xEncCoeffQT( tuRecurseChild, component, bRealCoeff );
    } while (tuRecurseChild.nextSection(rTu) );
  }
  else if (rTu.ProcessComponentSection(component))
  {
    //===== coefficients =====
    const UInt  uiLog2TrafoSize = rTu.GetLog2LumaTrSize();
    UInt    uiCoeffOffset   = rTu.getCoefficientOffset(component);
    UInt    uiQTLayer       = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrafoSize;
    TCoeff* pcCoeff         = bRealCoeff ? pcCU->getCoeff(component) : m_ppcQTTempCoeff[component][uiQTLayer];

    if (isChroma(component) && (pcCU->getCbf( rTu.GetAbsPartIdxTU(), COMPONENT_Y, uiTrMode ) != 0) && pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag() )
    {
      m_pcEntropyCoder->encodeCrossComponentPrediction( rTu, component );
    }

    m_pcEntropyCoder->encodeCoeffNxN( rTu, pcCoeff+uiCoeffOffset, component );
  }
}




Void
TEncSearch::xEncIntraHeader( TComDataCU*  pcCU,
                            UInt         uiTrDepth,
                            UInt         uiAbsPartIdx,
                            Bool         bLuma,
                            Bool         bChroma )
{
  if( bLuma )
  {
    // CU header
    if( uiAbsPartIdx == 0 )
    {
      if( !pcCU->getSlice()->isIntra() )
      {
        if (pcCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
        {
          m_pcEntropyCoder->encodeCUTransquantBypassFlag( pcCU, 0, true );
        }
        m_pcEntropyCoder->encodeSkipFlag( pcCU, 0, true );
        m_pcEntropyCoder->encodePredMode( pcCU, 0, true );
        m_pcEntropyCoder->encodePaletteModeInfo( pcCU, 0, true );
      }
      else // encodePredMode has already done it
      {
        if (pcCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
        {
          m_pcEntropyCoder->encodeCUTransquantBypassFlag(pcCU, 0, true);
        }
        m_pcEntropyCoder->encodePaletteModeInfo(pcCU, 0, true);
      }

      if (pcCU->getPaletteModeFlag(0))
      {
        return;
      }
      m_pcEntropyCoder  ->encodePartSize( pcCU, 0, pcCU->getDepth(0), true );

      if (pcCU->isIntra(0) && pcCU->getPartitionSize(0) == SIZE_2Nx2N )
      {
        m_pcEntropyCoder->encodeIPCMInfo( pcCU, 0, true );

        if ( pcCU->getIPCMFlag (0))
        {
          return;
        }
      }
    }
    // luma prediction mode
    if( pcCU->getPartitionSize(0) == SIZE_2Nx2N )
    {
      if (uiAbsPartIdx==0)
      {
        m_pcEntropyCoder->encodeIntraDirModeLuma ( pcCU, 0 );
      }
    }
    else
    {
      UInt uiQNumParts = pcCU->getTotalNumPart() >> 2;
      if (uiTrDepth>0 && (uiAbsPartIdx%uiQNumParts)==0)
      {
        m_pcEntropyCoder->encodeIntraDirModeLuma ( pcCU, uiAbsPartIdx );
      }
    }
  }

  if( bChroma )
  {
    if( pcCU->getPartitionSize(0) == SIZE_2Nx2N || !enable4ChromaPUsInIntraNxNCU(pcCU->getPic()->getChromaFormat()))
    {
      if(uiAbsPartIdx==0)
      {
         m_pcEntropyCoder->encodeIntraDirModeChroma ( pcCU, uiAbsPartIdx );
      }
    }
    else
    {
      UInt uiQNumParts = pcCU->getTotalNumPart() >> 2;
      assert(uiTrDepth>0);
      if ((uiAbsPartIdx%uiQNumParts)==0)
      {
        m_pcEntropyCoder->encodeIntraDirModeChroma ( pcCU, uiAbsPartIdx );
      }
    }
  }
}




UInt
TEncSearch::xGetIntraBitsQT(TComTU &rTu,
                            Bool         bLuma,
                            Bool         bChroma,
                            Bool         bRealCoeff /* just for test */ )
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiTrDepth=rTu.GetTransformDepthRel();
  m_pcEntropyCoder->resetBits();
  xEncIntraHeader ( pcCU, uiTrDepth, uiAbsPartIdx, bLuma, bChroma );
  xEncSubdivCbfQT ( rTu, bLuma, bChroma );

  if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && bLuma && bChroma )
  {
    xEncColorTransformFlagQT(rTu);
  }

  if( bLuma )
  {
    xEncCoeffQT   ( rTu, COMPONENT_Y,      bRealCoeff );
  }
  if( bChroma )
  {
    xEncCoeffQT   ( rTu, COMPONENT_Cb,  bRealCoeff );
    xEncCoeffQT   ( rTu, COMPONENT_Cr,  bRealCoeff );
  }
  UInt   uiBits = m_pcEntropyCoder->getNumberOfWrittenBits();

  return uiBits;
}

UInt TEncSearch::xGetIntraBitsQTChroma(TComTU &rTu,
                                       ComponentID compID,
                                       Bool         bRealCoeff /* just for test */ )
{
  m_pcEntropyCoder->resetBits();
  xEncCoeffQT   ( rTu, compID,  bRealCoeff );
  UInt   uiBits = m_pcEntropyCoder->getNumberOfWrittenBits();
  return uiBits;
}

Void TEncSearch::xIntraCodingTUBlock(       TComYuv*    pcOrgYuv,
                                            TComYuv*    pcPredYuv,
                                            TComYuv*    pcResiYuv,
                                            Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                      const Bool        checkCrossCPrediction,
                                            Distortion& ruiDist,
                                      const ComponentID compID,
                                            TComTU&     rTu
                                      DEBUG_STRING_FN_DECLARE(sDebug)
                                           ,Int         default0Save1Load2
                                     )
{
  if (!rTu.ProcessComponentSection(compID))
  {
    return;
  }
  const Bool           bIsLuma          = isLuma(compID);
  const TComRectangle &rect             = rTu.getRect(compID);
        TComDataCU    *pcCU             = rTu.getCU();
  const UInt           uiAbsPartIdx     = rTu.GetAbsPartIdxTU();
  const TComSPS       &sps              = *(pcCU->getSlice()->getSPS());

  const UInt           uiTrDepth        = rTu.GetTransformDepthRelAdj(compID);
  const UInt           uiFullDepth      = rTu.GetTransformDepthTotal();
  const UInt           uiLog2TrSize     = rTu.GetLog2LumaTrSize();
  const ChromaFormat   chFmt            = pcOrgYuv->getChromaFormat();
  const ChannelType    chType           = toChannelType(compID);
  const Int            bitDepth         = sps.getBitDepth(chType);

  const UInt           uiWidth          = rect.width;
  const UInt           uiHeight         = rect.height;
  const UInt           uiStride         = pcOrgYuv ->getStride (compID);
        Pel           *piOrg            = pcOrgYuv ->getAddr( compID, uiAbsPartIdx );
        Pel           *piPred           = pcPredYuv->getAddr( compID, uiAbsPartIdx );
        Pel           *piResi           = pcResiYuv->getAddr( compID, uiAbsPartIdx );
        Pel           *piReco           = pcPredYuv->getAddr( compID, uiAbsPartIdx );
  const UInt           uiQTLayer        = sps.getQuadtreeTULog2MaxSize() - uiLog2TrSize;
        Pel           *piRecQt          = m_pcQTTempTComYuv[ uiQTLayer ].getAddr( compID, uiAbsPartIdx );
  const UInt           uiRecQtStride    = m_pcQTTempTComYuv[ uiQTLayer ].getStride(compID);
  const UInt           uiZOrder         = pcCU->getZorderIdxInCtu() + uiAbsPartIdx;
        Pel           *piRecIPred       = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), uiZOrder );
        UInt           uiRecIPredStride = pcCU->getPic()->getPicYuvRec()->getStride  ( compID );
        TCoeff        *pcCoeff          = m_ppcQTTempCoeff[compID][uiQTLayer] + rTu.getCoefficientOffset(compID);
        Bool           useTransformSkip = pcCU->getTransformSkip(uiAbsPartIdx, compID);

#if ADAPTIVE_QP_SELECTION
        TCoeff        *pcArlCoeff       = m_ppcQTTempArlCoeff[compID][ uiQTLayer ] + rTu.getCoefficientOffset(compID);
#endif

  const UInt           uiChPredMode     = pcCU->getIntraDir( chType, uiAbsPartIdx );
  const UInt           partsPerMinCU    = 1<<(2*(sps.getMaxTotalCUDepth() - sps.getLog2DiffMaxMinCodingBlockSize()));
  const UInt           uiChCodedMode    = (uiChPredMode==DM_CHROMA_IDX && !bIsLuma) ? pcCU->getIntraDir(CHANNEL_TYPE_LUMA, getChromasCorrespondingPULumaIdx(uiAbsPartIdx, chFmt, partsPerMinCU)) : uiChPredMode;
  const UInt           uiChFinalMode    = ((chFmt == CHROMA_422)       && !bIsLuma) ? g_chroma422IntraAngleMappingTable[uiChCodedMode] : uiChCodedMode;

  const Int            blkX                                 = g_auiRasterToPelX[ g_auiZscanToRaster[ uiAbsPartIdx ] ];
  const Int            blkY                                 = g_auiRasterToPelY[ g_auiZscanToRaster[ uiAbsPartIdx ] ];
  const Int            bufferOffset                         = blkX + (blkY * MAX_CU_SIZE);
        Pel  *const    encoderLumaResidual                  = resiLuma[RESIDUAL_ENCODER_SIDE ] + bufferOffset;
        Pel  *const    reconstructedLumaResidual            = resiLuma[RESIDUAL_RECONSTRUCTED] + bufferOffset;
  const Bool           bUseCrossCPrediction                 = isChroma(compID) && (uiChPredMode == DM_CHROMA_IDX) && checkCrossCPrediction;
  const Bool           bUseReconstructedResidualForEstimate = m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate();
        Pel *const     lumaResidualForEstimate              = bUseReconstructedResidualForEstimate ? reconstructedLumaResidual : encoderLumaResidual;

#if DEBUG_STRING
  const Int debugPredModeMask=DebugStringGetPredModeMask(MODE_INTRA);
#endif

  assert(!pcCU->getColourTransform(uiAbsPartIdx));
  QpParam cQP(*pcCU, compID, uiAbsPartIdx);

  //===== init availability pattern =====
  DEBUG_STRING_NEW(sTemp)

#if !DEBUG_STRING
  if( default0Save1Load2 != 2 )
#endif
  {
    const Bool bUseFilteredPredictions=TComPrediction::filteringIntraReferenceSamples(compID, uiChFinalMode, uiWidth, uiHeight, chFmt, sps.getSpsRangeExtension().getIntraSmoothingDisabledFlag());

    initIntraPatternChType( rTu, compID, bUseFilteredPredictions DEBUG_STRING_PASS_INTO(sDebug) );

    //===== get prediction signal =====
    predIntraAng( compID, uiChFinalMode, piOrg, uiStride, piPred, uiStride, rTu, bUseFilteredPredictions );

    // save prediction
    if( default0Save1Load2 == 1 )
    {
      Pel*  pPred   = piPred;
      Pel*  pPredBuf = m_pSharedPredTransformSkip[compID];
      Int k = 0;
      for( UInt uiY = 0; uiY < uiHeight; uiY++ )
      {
        for( UInt uiX = 0; uiX < uiWidth; uiX++ )
        {
          pPredBuf[ k ++ ] = pPred[ uiX ];
        }
        pPred += uiStride;
      }
    }
  }
#if !DEBUG_STRING
  else
  {
    // load prediction
    Pel*  pPred   = piPred;
    Pel*  pPredBuf = m_pSharedPredTransformSkip[compID];
    Int k = 0;
    for( UInt uiY = 0; uiY < uiHeight; uiY++ )
    {
      for( UInt uiX = 0; uiX < uiWidth; uiX++ )
      {
        pPred[ uiX ] = pPredBuf[ k ++ ];
      }
      pPred += uiStride;
    }
  }
#endif

  //===== get residual signal =====
  {
    // get residual
    Pel*  pOrg    = piOrg;
    Pel*  pPred   = piPred;
    Pel*  pResi   = piResi;

    for( UInt uiY = 0; uiY < uiHeight; uiY++ )
    {
      for( UInt uiX = 0; uiX < uiWidth; uiX++ )
      {
        pResi[ uiX ] = pOrg[ uiX ] - pPred[ uiX ];
      }

      pOrg  += uiStride;
      pResi += uiStride;
      pPred += uiStride;
    }
  }

  if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
  {
    if (bUseCrossCPrediction)
    {
      if (xCalcCrossComponentPredictionAlpha( rTu, compID, lumaResidualForEstimate, piResi, uiWidth, uiHeight, MAX_CU_SIZE, uiStride ) == 0)
      {
        return;
      }
      TComTrQuant::crossComponentPrediction ( rTu, compID, reconstructedLumaResidual, piResi, piResi, uiWidth, uiHeight, MAX_CU_SIZE, uiStride, uiStride, false );
    }
    else if (isLuma(compID) && !bUseReconstructedResidualForEstimate)
    {
      xStoreCrossComponentPredictionResult( encoderLumaResidual, piResi, rTu, 0, 0, MAX_CU_SIZE, uiStride );
    }
  }

  //===== transform and quantization =====
  //--- init rate estimation arrays for RDOQ ---
  if( useTransformSkip ? m_pcEncCfg->getUseRDOQTS() : m_pcEncCfg->getUseRDOQ() )
  {
    COEFF_SCAN_TYPE scanType = COEFF_SCAN_TYPE(pcCU->getCoefScanIdx(uiAbsPartIdx, uiWidth, uiHeight, compID));
    m_pcEntropyCoder->estimateBit( m_pcTrQuant->m_pcEstBitsSbac, uiWidth, uiHeight, chType, scanType );
  }

  //--- transform and quantization ---
  TCoeff uiAbsSum = 0;
  if (bIsLuma)
  {
    pcCU       ->setTrIdxSubParts ( uiTrDepth, uiAbsPartIdx, uiFullDepth );
  }

#if RDOQ_CHROMA_LAMBDA
  m_pcTrQuant->selectLambda     (compID);
#endif

  m_pcTrQuant->transformNxN     ( rTu, compID, piResi, uiStride, pcCoeff,
#if ADAPTIVE_QP_SELECTION
    pcArlCoeff,
#endif
    uiAbsSum, cQP
    );

  //--- inverse transform ---

#if DEBUG_STRING
  if ( (uiAbsSum > 0) || (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask) )
#else
  if ( uiAbsSum > 0 )
#endif
  {
    m_pcTrQuant->invTransformNxN ( rTu, compID, piResi, uiStride, pcCoeff, cQP DEBUG_STRING_PASS_INTO_OPTIONAL(&sDebug, (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask)) );
  }
  else
  {
    Pel* pResi = piResi;
    memset( pcCoeff, 0, sizeof( TCoeff ) * uiWidth * uiHeight );
    for( UInt uiY = 0; uiY < uiHeight; uiY++ )
    {
      memset( pResi, 0, sizeof( Pel ) * uiWidth );
      pResi += uiStride;
    }
  }


  //===== reconstruction =====
  {
    Pel* pPred      = piPred;
    Pel* pResi      = piResi;
    Pel* pReco      = piReco;
    Pel* pRecQt     = piRecQt;
    Pel* pRecIPred  = piRecIPred;

    if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
    {
      if (bUseCrossCPrediction)
      {
        TComTrQuant::crossComponentPrediction( rTu, compID, reconstructedLumaResidual, piResi, piResi, uiWidth, uiHeight, MAX_CU_SIZE, uiStride, uiStride, true );
      }
      else if (isLuma(compID))
      {
        xStoreCrossComponentPredictionResult( reconstructedLumaResidual, piResi, rTu, 0, 0, MAX_CU_SIZE, uiStride );
      }
    }

 #if DEBUG_STRING
    std::stringstream ss(stringstream::out);
    const Bool bDebugPred=((DebugOptionList::DebugString_Pred.getInt()&debugPredModeMask) && DEBUG_STRING_CHANNEL_CONDITION(compID));
    const Bool bDebugResi=((DebugOptionList::DebugString_Resi.getInt()&debugPredModeMask) && DEBUG_STRING_CHANNEL_CONDITION(compID));
    const Bool bDebugReco=((DebugOptionList::DebugString_Reco.getInt()&debugPredModeMask) && DEBUG_STRING_CHANNEL_CONDITION(compID));

    if (bDebugPred || bDebugResi || bDebugReco)
    {
      ss << "###: " << "CompID: " << compID << " pred mode (ch/fin): " << uiChPredMode << "/" << uiChFinalMode << " absPartIdx: " << rTu.GetAbsPartIdxTU() << "\n";
      for( UInt uiY = 0; uiY < uiHeight; uiY++ )
      {
        ss << "###: ";
        if (bDebugPred)
        {
          ss << " - pred: ";
          for( UInt uiX = 0; uiX < uiWidth; uiX++ )
          {
            ss << pPred[ uiX ] << ", ";
          }
        }
        if (bDebugResi)
        {
          ss << " - resi: ";
        }
        for( UInt uiX = 0; uiX < uiWidth; uiX++ )
        {
          if (bDebugResi)
          {
            ss << pResi[ uiX ] << ", ";
          }
          pReco    [ uiX ] = Pel(ClipBD<Int>( Int(pPred[uiX]) + Int(pResi[uiX]), bitDepth ));
          pRecQt   [ uiX ] = pReco[ uiX ];
          pRecIPred[ uiX ] = pReco[ uiX ];
        }
        if (bDebugReco)
        {
          ss << " - reco: ";
          for( UInt uiX = 0; uiX < uiWidth; uiX++ )
          {
            ss << pReco[ uiX ] << ", ";
          }
        }
        pPred     += uiStride;
        pResi     += uiStride;
        pReco     += uiStride;
        pRecQt    += uiRecQtStride;
        pRecIPred += uiRecIPredStride;
        ss << "\n";
      }
      DEBUG_STRING_APPEND(sDebug, ss.str())
    }
    else
#endif
    {

      for( UInt uiY = 0; uiY < uiHeight; uiY++ )
      {
        for( UInt uiX = 0; uiX < uiWidth; uiX++ )
        {
          pReco    [ uiX ] = Pel(ClipBD<Int>( Int(pPred[uiX]) + Int(pResi[uiX]), bitDepth ));
          pRecQt   [ uiX ] = pReco[ uiX ];
          pRecIPred[ uiX ] = pReco[ uiX ];
        }
        pPred     += uiStride;
        pResi     += uiStride;
        pReco     += uiStride;
        pRecQt    += uiRecQtStride;
        pRecIPred += uiRecIPredStride;
      }
    }
  }

  //===== update distortion =====
  ruiDist += m_pcRdCost->getDistPart( bitDepth, piReco, uiStride, piOrg, uiStride, uiWidth, uiHeight, compID );
}




Void
TEncSearch::xRecurIntraCodingLumaQT(TComYuv*    pcOrgYuv,
                                    TComYuv*    pcPredYuv,
                                    TComYuv*    pcResiYuv,
                                    Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                    Distortion& ruiDistY,
#if HHI_RQT_INTRA_SPEEDUP
                                    Bool        bCheckFirst,
#endif
                                    Double&     dRDCost,
                                    TComTU&     rTu
                                    DEBUG_STRING_FN_DECLARE(sDebug))
{
  TComDataCU   *pcCU          = rTu.getCU();
  const UInt    uiAbsPartIdx  = rTu.GetAbsPartIdxTU();
  const UInt    uiFullDepth   = rTu.GetTransformDepthTotal();
  const UInt    uiTrDepth     = rTu.GetTransformDepthRel();
  const UInt    uiLog2TrSize  = rTu.GetLog2LumaTrSize();
        Bool    bCheckFull    = ( uiLog2TrSize  <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() );
        Bool    bCheckSplit   = ( uiLog2TrSize  >  pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) );

        Pel     resiLumaSplit [NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];
        Pel     resiLumaSingle[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];

        Bool    bMaintainResidual[NUMBER_OF_STORED_RESIDUAL_TYPES];
        for (UInt residualTypeIndex = 0; residualTypeIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; residualTypeIndex++)
        {
          bMaintainResidual[residualTypeIndex] = true; //assume true unless specified otherwise
        }

        bMaintainResidual[RESIDUAL_ENCODER_SIDE] = !(m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate());

#if HHI_RQT_INTRA_SPEEDUP
  Int maxTuSize = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize();
  Int isIntraSlice = (pcCU->getSlice()->getSliceType() == I_SLICE);
  // don't check split if TU size is less or equal to max TU size
  Bool noSplitIntraMaxTuSize = bCheckFull;
  if(m_pcEncCfg->getRDpenalty() && ! isIntraSlice)
  {
    // in addition don't check split if TU size is less or equal to 16x16 TU size for non-intra slice
    noSplitIntraMaxTuSize = ( uiLog2TrSize  <= min(maxTuSize,4) );

    // if maximum RD-penalty don't check TU size 32x32
    if(m_pcEncCfg->getRDpenalty()==2)
    {
      bCheckFull    = ( uiLog2TrSize  <= min(maxTuSize,4));
    }
  }
  if( bCheckFirst && noSplitIntraMaxTuSize )

  {
    bCheckSplit = false;
  }
#else
  Int maxTuSize = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize();
  Int isIntraSlice = (pcCU->getSlice()->getSliceType() == I_SLICE);
  // if maximum RD-penalty don't check TU size 32x32
  if((m_pcEncCfg->getRDpenalty()==2)  && !isIntraSlice)
  {
    bCheckFull    = ( uiLog2TrSize  <= min(maxTuSize,4));
  }
#endif

  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(uiAbsPartIdx) && bCheckFull)
  {
    bCheckSplit = false;
  }

  Double     dSingleCost                        = MAX_DOUBLE;
  Distortion uiSingleDistLuma                   = 0;
  UInt       uiSingleCbfLuma                    = 0;
  Bool       checkTransformSkip  = pcCU->getSlice()->getPPS()->getUseTransformSkip();
  Int        bestModeId[MAX_NUM_COMPONENT] = { 0, 0, 0};
  checkTransformSkip           &= TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(COMPONENT_Y), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize());
  checkTransformSkip           &= (!pcCU->getCUTransquantBypass(0));

  assert (rTu.ProcessComponentSection(COMPONENT_Y));
  const UInt totalAdjustedDepthChan   = rTu.GetTransformDepthTotalAdj(COMPONENT_Y);

  if ( m_pcEncCfg->getUseTransformSkipFast() )
  {
    checkTransformSkip       &= (pcCU->getPartitionSize(uiAbsPartIdx)==SIZE_NxN);
  }

  if( bCheckFull )
  {
    if(checkTransformSkip == true)
    {
      //----- store original entropy coding status -----
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );

      Distortion singleDistTmpLuma                    = 0;
      UInt       singleCbfTmpLuma                     = 0;
      Double     singleCostTmp                        = 0;
      Int        firstCheckId                         = 0;

      for(Int modeId = firstCheckId; modeId < 2; modeId ++)
      {
        DEBUG_STRING_NEW(sModeString)
        Int  default0Save1Load2 = 0;
        singleDistTmpLuma=0;
        if(modeId == firstCheckId)
        {
          default0Save1Load2 = 1;
        }
        else
        {
          default0Save1Load2 = 2;
        }


        pcCU->setTransformSkipSubParts ( modeId, COMPONENT_Y, uiAbsPartIdx, totalAdjustedDepthChan );
        xIntraCodingTUBlock( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaSingle, false, singleDistTmpLuma, COMPONENT_Y, rTu DEBUG_STRING_PASS_INTO(sModeString), default0Save1Load2 );

        singleCbfTmpLuma = pcCU->getCbf( uiAbsPartIdx, COMPONENT_Y, uiTrDepth );

        //----- determine rate and r-d cost -----
        if(modeId == 1 && singleCbfTmpLuma == 0)
        {
          //In order not to code TS flag when cbf is zero, the case for TS with cbf being zero is forbidden.
          singleCostTmp = MAX_DOUBLE;
        }
        else
        {
          UInt uiSingleBits = xGetIntraBitsQT( rTu, true, false, false );
          singleCostTmp     = m_pcRdCost->calcRdCost( uiSingleBits, singleDistTmpLuma );
        }
        if(singleCostTmp < dSingleCost)
        {
          DEBUG_STRING_SWAP(sDebug, sModeString)
          dSingleCost   = singleCostTmp;
          uiSingleDistLuma = singleDistTmpLuma;
          uiSingleCbfLuma = singleCbfTmpLuma;

          bestModeId[COMPONENT_Y] = modeId;
          if(bestModeId[COMPONENT_Y] == firstCheckId)
          {
            xStoreIntraResultQT(COMPONENT_Y, rTu );
            m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_TEMP_BEST ] );
          }

          if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
          {
            const Int xOffset = rTu.getRect( COMPONENT_Y ).x0;
            const Int yOffset = rTu.getRect( COMPONENT_Y ).y0;
            for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
            {
              if (bMaintainResidual[storedResidualIndex])
              {
                xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaSingle[storedResidualIndex], rTu, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE);
              }
            }
          }
        }
        if (modeId == firstCheckId)
        {
          m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
        }
      }

      pcCU ->setTransformSkipSubParts ( bestModeId[COMPONENT_Y], COMPONENT_Y, uiAbsPartIdx, totalAdjustedDepthChan );

      if(bestModeId[COMPONENT_Y] == firstCheckId)
      {
        xLoadIntraResultQT(COMPONENT_Y, rTu );
        pcCU->setCbfSubParts  ( uiSingleCbfLuma << uiTrDepth, COMPONENT_Y, uiAbsPartIdx, rTu.GetTransformDepthTotalAdj(COMPONENT_Y) );

        m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiFullDepth ][ CI_TEMP_BEST ] );
      }
    }
    else
    {
      //----- store original entropy coding status -----
      if( bCheckSplit )
      {
        m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
      }
      //----- code luma/chroma block with given intra prediction mode and store Cbf-----
      dSingleCost   = 0.0;

      pcCU ->setTransformSkipSubParts ( 0, COMPONENT_Y, uiAbsPartIdx, totalAdjustedDepthChan );
      xIntraCodingTUBlock( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaSingle, false, uiSingleDistLuma, COMPONENT_Y, rTu DEBUG_STRING_PASS_INTO(sDebug));

      if( bCheckSplit )
      {
        uiSingleCbfLuma = pcCU->getCbf( uiAbsPartIdx, COMPONENT_Y, uiTrDepth );
      }
      //----- determine rate and r-d cost -----
      UInt uiSingleBits = xGetIntraBitsQT( rTu, true, false, false );

      if(m_pcEncCfg->getRDpenalty() && (uiLog2TrSize==5) && !isIntraSlice)
      {
        uiSingleBits=uiSingleBits*4;
      }

      dSingleCost       = m_pcRdCost->calcRdCost( uiSingleBits, uiSingleDistLuma );

      if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
      {
        const Int xOffset = rTu.getRect( COMPONENT_Y ).x0;
        const Int yOffset = rTu.getRect( COMPONENT_Y ).y0;
        for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
        {
          if (bMaintainResidual[storedResidualIndex])
          {
            xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaSingle[storedResidualIndex], rTu, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE);
          }
        }
      }
    }
  }

  if( bCheckSplit )
  {
    //----- store full entropy coding status, load original entropy coding status -----
    if( bCheckFull )
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_TEST ] );
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
    }
    else
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
    }
    //----- code splitted block -----
    Double     dSplitCost      = 0.0;
    Distortion uiSplitDistLuma = 0;
    UInt       uiSplitCbfLuma  = 0;

    TComTURecurse tuRecurseChild(rTu, false);
    DEBUG_STRING_NEW(sSplit)
    do
    {
      DEBUG_STRING_NEW(sChild)
#if HHI_RQT_INTRA_SPEEDUP
      xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaSplit, uiSplitDistLuma, bCheckFirst, dSplitCost, tuRecurseChild DEBUG_STRING_PASS_INTO(sChild) );
#else
      xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaSplit, uiSplitDistLuma, dSplitCost, tuRecurseChild DEBUG_STRING_PASS_INTO(sChild) );
#endif
      DEBUG_STRING_APPEND(sSplit, sChild)
      uiSplitCbfLuma |= pcCU->getCbf( tuRecurseChild.GetAbsPartIdxTU(), COMPONENT_Y, tuRecurseChild.GetTransformDepthRel() );
    } while (tuRecurseChild.nextSection(rTu) );

    UInt    uiPartsDiv     = rTu.GetAbsPartIdxNumParts();
    {
      if (uiSplitCbfLuma)
      {
        const UInt flag=1<<uiTrDepth;
        UChar *pBase=pcCU->getCbf( COMPONENT_Y );
        for( UInt uiOffs = 0; uiOffs < uiPartsDiv; uiOffs++ )
        {
          pBase[ uiAbsPartIdx + uiOffs ] |= flag;
        }
      }
    }
    //----- restore context states -----
    m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
    
    //----- determine rate and r-d cost -----
    UInt uiSplitBits = xGetIntraBitsQT( rTu, true, false, false );
    dSplitCost       = m_pcRdCost->calcRdCost( uiSplitBits, uiSplitDistLuma );

    //===== compare and set best =====
    if( dSplitCost < dSingleCost )
    {
      //--- update cost ---
      DEBUG_STRING_SWAP(sSplit, sDebug)
      ruiDistY += uiSplitDistLuma;
      dRDCost  += dSplitCost;

      if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
      {
        const Int xOffset = rTu.getRect( COMPONENT_Y ).x0;
        const Int yOffset = rTu.getRect( COMPONENT_Y ).y0;
        for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
        {
          if (bMaintainResidual[storedResidualIndex])
          {
            xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaSplit[storedResidualIndex], rTu, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE);
          }
        }
      }

      return;
    }

    //----- set entropy coding status -----
    m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_TEST ] );

    //--- set transform index and Cbf values ---
    pcCU->setTrIdxSubParts( uiTrDepth, uiAbsPartIdx, uiFullDepth );
    const TComRectangle &tuRect=rTu.getRect(COMPONENT_Y);
    pcCU->setCbfSubParts  ( uiSingleCbfLuma << uiTrDepth, COMPONENT_Y, uiAbsPartIdx, totalAdjustedDepthChan );
    pcCU ->setTransformSkipSubParts  ( bestModeId[COMPONENT_Y], COMPONENT_Y, uiAbsPartIdx, totalAdjustedDepthChan );

    //--- set reconstruction for next intra prediction blocks ---
    const UInt  uiQTLayer   = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
    const UInt  uiZOrder    = pcCU->getZorderIdxInCtu() + uiAbsPartIdx;
    const UInt  uiWidth     = tuRect.width;
    const UInt  uiHeight    = tuRect.height;
    Pel*  piSrc       = m_pcQTTempTComYuv[ uiQTLayer ].getAddr( COMPONENT_Y, uiAbsPartIdx );
    UInt  uiSrcStride = m_pcQTTempTComYuv[ uiQTLayer ].getStride  ( COMPONENT_Y );
    Pel*  piDes       = pcCU->getPic()->getPicYuvRec()->getAddr( COMPONENT_Y, pcCU->getCtuRsAddr(), uiZOrder );
    UInt  uiDesStride = pcCU->getPic()->getPicYuvRec()->getStride  ( COMPONENT_Y );

    for( UInt uiY = 0; uiY < uiHeight; uiY++, piSrc += uiSrcStride, piDes += uiDesStride )
    {
      for( UInt uiX = 0; uiX < uiWidth; uiX++ )
      {
        piDes[ uiX ] = piSrc[ uiX ];
      }
    }
  }
  ruiDistY += uiSingleDistLuma;
  dRDCost  += dSingleCost;
}


Void
TEncSearch::xSetIntraResultLumaQT(TComYuv* pcRecoYuv, TComTU &rTu)
{
  TComDataCU *pcCU        = rTu.getCU();
  const UInt uiTrDepth    = rTu.GetTransformDepthRel();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  UInt uiTrMode     = pcCU->getTransformIdx( uiAbsPartIdx );
  if(  uiTrMode == uiTrDepth )
  {
    UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();
    UInt uiQTLayer    = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;

    //===== copy transform coefficients =====

    const TComRectangle &tuRect=rTu.getRect(COMPONENT_Y);
    const UInt coeffOffset = rTu.getCoefficientOffset(COMPONENT_Y);
    const UInt numCoeffInBlock = tuRect.width * tuRect.height;

    if (numCoeffInBlock!=0)
    {
      const TCoeff* srcCoeff = m_ppcQTTempCoeff[COMPONENT_Y][uiQTLayer] + coeffOffset;
      TCoeff* destCoeff      = pcCU->getCoeff(COMPONENT_Y) + coeffOffset;
      ::memcpy( destCoeff, srcCoeff, sizeof(TCoeff)*numCoeffInBlock );
#if ADAPTIVE_QP_SELECTION
      const TCoeff* srcArlCoeff = m_ppcQTTempArlCoeff[COMPONENT_Y][ uiQTLayer ] + coeffOffset;
      TCoeff* destArlCoeff      = pcCU->getArlCoeff (COMPONENT_Y)               + coeffOffset;
      ::memcpy( destArlCoeff, srcArlCoeff, sizeof( TCoeff ) * numCoeffInBlock );
#endif
      m_pcQTTempTComYuv[ uiQTLayer ].copyPartToPartComponent( COMPONENT_Y, pcRecoYuv, uiAbsPartIdx, tuRect.width, tuRect.height );
    }

  }
  else
  {
    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xSetIntraResultLumaQT( pcRecoYuv, tuRecurseChild );
    } while (tuRecurseChild.nextSection(rTu));
  }
}


Void
TEncSearch::xStoreIntraResultQT(const ComponentID compID, TComTU &rTu, Bool bACTCache )
{
  TComDataCU *pcCU=rTu.getCU();
  const UInt uiTrDepth = rTu.GetTransformDepthRel();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiTrMode     = pcCU->getTransformIdx( uiAbsPartIdx );
  if ( compID==COMPONENT_Y || uiTrMode == uiTrDepth )
  {
    assert(uiTrMode == uiTrDepth);
    const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();
    const UInt uiQTLayer    = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;

    if (rTu.ProcessComponentSection(compID))
    {
      const TComRectangle &tuRect=rTu.getRect(compID);

      //===== copy transform coefficients =====
      const UInt uiNumCoeff    = tuRect.width * tuRect.height;
      TCoeff* pcCoeffSrc = m_ppcQTTempCoeff[compID] [ uiQTLayer ] + rTu.getCoefficientOffset(compID);
      TCoeff* pcCoeffDst = (!bACTCache)? m_pcQTTempTUCoeff[compID]: m_pcACTTempTUCoeff[compID];

      ::memcpy( pcCoeffDst, pcCoeffSrc, sizeof( TCoeff ) * uiNumCoeff );
#if ADAPTIVE_QP_SELECTION
      TCoeff* pcArlCoeffSrc = m_ppcQTTempArlCoeff[compID] [ uiQTLayer ] + rTu.getCoefficientOffset(compID);
      TCoeff* pcArlCoeffDst = (!bACTCache)? m_ppcQTTempTUArlCoeff[compID]: m_ppcACTTempTUArlCoeff[compID];
      ::memcpy( pcArlCoeffDst, pcArlCoeffSrc, sizeof( TCoeff ) * uiNumCoeff );
#endif
      //===== copy reconstruction =====
      m_pcQTTempTComYuv[ uiQTLayer ].copyPartToPartComponent( compID, ((!bACTCache)? &m_pcQTTempTransformSkipTComYuv: &m_pcACTTempTransformSkipTComYuv), uiAbsPartIdx, tuRect.width, tuRect.height );
    }
  }
}

Void
TEncSearch::xLoadIntraResultQT(const ComponentID compID, TComTU &rTu, Bool bACTCache)
{
  TComDataCU *pcCU=rTu.getCU();
  const UInt uiTrDepth = rTu.GetTransformDepthRel();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiTrMode     = pcCU->getTransformIdx( uiAbsPartIdx );
  if ( compID==COMPONENT_Y || uiTrMode == uiTrDepth )
  {
    assert(uiTrMode == uiTrDepth);
    const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();
    const UInt uiQTLayer    = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
    const UInt uiZOrder     = pcCU->getZorderIdxInCtu() + uiAbsPartIdx;

    if (rTu.ProcessComponentSection(compID))
    {
      const TComRectangle &tuRect=rTu.getRect(compID);

      //===== copy transform coefficients =====
      const UInt uiNumCoeff = tuRect.width * tuRect.height;
      TCoeff* pcCoeffDst = m_ppcQTTempCoeff[compID] [ uiQTLayer ] + rTu.getCoefficientOffset(compID);
      TCoeff* pcCoeffSrc = (!bACTCache)?m_pcQTTempTUCoeff[compID]: m_pcACTTempTUCoeff[compID];

      ::memcpy( pcCoeffDst, pcCoeffSrc, sizeof( TCoeff ) * uiNumCoeff );
#if ADAPTIVE_QP_SELECTION
      TCoeff* pcArlCoeffDst = m_ppcQTTempArlCoeff[compID] [ uiQTLayer ] + rTu.getCoefficientOffset(compID);
      TCoeff* pcArlCoeffSrc = (!bACTCache)? m_ppcQTTempTUArlCoeff[compID]: m_ppcACTTempTUArlCoeff[compID];
      ::memcpy( pcArlCoeffDst, pcArlCoeffSrc, sizeof( TCoeff ) * uiNumCoeff );
#endif
      //===== copy reconstruction =====
      if( !bACTCache )
      {
        m_pcQTTempTransformSkipTComYuv.copyPartToPartComponent( compID, &m_pcQTTempTComYuv[ uiQTLayer ], uiAbsPartIdx, tuRect.width, tuRect.height );
      }
      else
      {
        m_pcACTTempTransformSkipTComYuv.copyPartToPartComponent( compID, &m_pcQTTempTComYuv[ uiQTLayer ], uiAbsPartIdx, tuRect.width, tuRect.height );
      }

      Pel*    piRecIPred        = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), uiZOrder );
      UInt    uiRecIPredStride  = pcCU->getPic()->getPicYuvRec()->getStride (compID);
      Pel*    piRecQt           = m_pcQTTempTComYuv[ uiQTLayer ].getAddr( compID, uiAbsPartIdx );
      UInt    uiRecQtStride     = m_pcQTTempTComYuv[ uiQTLayer ].getStride  (compID);
      UInt    uiWidth           = tuRect.width;
      UInt    uiHeight          = tuRect.height;
      Pel* pRecQt               = piRecQt;
      Pel* pRecIPred            = piRecIPred;
      for( UInt uiY = 0; uiY < uiHeight; uiY++ )
      {
        for( UInt uiX = 0; uiX < uiWidth; uiX++ )
        {
          pRecIPred[ uiX ] = pRecQt   [ uiX ];
        }
        pRecQt    += uiRecQtStride;
        pRecIPred += uiRecIPredStride;
      }
    }
  }
}

Void
TEncSearch::xStoreCrossComponentPredictionResult(       Pel    *pResiDst,
                                                  const Pel    *pResiSrc,
                                                        TComTU &rTu,
                                                  const Int     xOffset,
                                                  const Int     yOffset,
                                                  const Int     strideDst,
                                                  const Int     strideSrc )
{
  const Pel *pSrc = pResiSrc + yOffset * strideSrc + xOffset;
        Pel *pDst = pResiDst + yOffset * strideDst + xOffset;

  for( Int y = 0; y < rTu.getRect( COMPONENT_Y ).height; y++ )
  {
    ::memcpy( pDst, pSrc, sizeof(Pel) * rTu.getRect( COMPONENT_Y ).width );
    pDst += strideDst;
    pSrc += strideSrc;
  }
}

SChar
TEncSearch::xCalcCrossComponentPredictionAlpha(       TComTU &rTu,
                                                const ComponentID compID,
                                                const Pel*        piResiL,
                                                const Pel*        piResiC,
                                                const Int         width,
                                                const Int         height,
                                                const Int         strideL,
                                                const Int         strideC )
{
  const Pel *pResiL = piResiL;
  const Pel *pResiC = piResiC;

        TComDataCU *pCU = rTu.getCU();
  const Int  absPartIdx = rTu.GetAbsPartIdxTU( compID );
  const Int diffBitDepth = pCU->getSlice()->getSPS()->getDifferentialLumaChromaBitDepth();

  SChar alpha = 0;
  Int SSxy  = 0;
  Int SSxx  = 0;

  for( UInt uiY = 0; uiY < height; uiY++ )
  {
    for( UInt uiX = 0; uiX < width; uiX++ )
    {
      const Pel scaledResiL = rightShift( pResiL[ uiX ], diffBitDepth );
      SSxy += ( scaledResiL * pResiC[ uiX ] );
      SSxx += ( scaledResiL * scaledResiL   );
    }

    pResiL += strideL;
    pResiC += strideC;
  }

  if( SSxx != 0 )
  {
    Double dAlpha = SSxy / Double( SSxx );
    alpha = SChar(Clip3<Int>(-16, 16, (Int)(dAlpha * 16)));

    static const SChar alphaQuant[17] = {0, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8};

    alpha = (alpha < 0) ? -alphaQuant[Int(-alpha)] : alphaQuant[Int(alpha)];
  }
  pCU->setCrossComponentPredictionAlphaPartRange( alpha, compID, absPartIdx, rTu.GetAbsPartIdxNumParts( compID ) );

  return alpha;
}

Void
TEncSearch::xRecurIntraChromaCodingQT(TComYuv*    pcOrgYuv,
                                      TComYuv*    pcPredYuv,
                                      TComYuv*    pcResiYuv,
                                      Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE],
                                      Distortion& ruiDist,
                                      TComTU&     rTu
                                      DEBUG_STRING_FN_DECLARE(sDebug))
{
  TComDataCU         *pcCU                  = rTu.getCU();
  const UInt          uiTrDepth             = rTu.GetTransformDepthRel();
  const UInt          uiAbsPartIdx          = rTu.GetAbsPartIdxTU();
  const ChromaFormat  format                = rTu.GetChromaFormat();
  UInt                uiTrMode              = pcCU->getTransformIdx( uiAbsPartIdx );
  const UInt          numberValidComponents = getNumberValidComponents(format);

  if(  uiTrMode == uiTrDepth )
  {
    if (!rTu.ProcessChannelSection(CHANNEL_TYPE_CHROMA))
    {
      return;
    }

    const UInt uiFullDepth = rTu.GetTransformDepthTotal();

    Bool checkTransformSkip = pcCU->getSlice()->getPPS()->getUseTransformSkip();
    checkTransformSkip &= TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(COMPONENT_Cb), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize());

    if ( m_pcEncCfg->getUseTransformSkipFast() )
    {
      checkTransformSkip &= TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(COMPONENT_Y), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize());

      if (checkTransformSkip)
      {
        Int nbLumaSkip = 0;
        const UInt maxAbsPartIdxSub=uiAbsPartIdx + (rTu.ProcessingAllQuadrants(COMPONENT_Cb)?1:4);
        for(UInt absPartIdxSub = uiAbsPartIdx; absPartIdxSub < maxAbsPartIdxSub; absPartIdxSub ++)
        {
          nbLumaSkip += pcCU->getTransformSkip(absPartIdxSub, COMPONENT_Y);
        }
        checkTransformSkip &= (nbLumaSkip > 0);
      }
    }


    for (UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
    {
      const ComponentID compID = ComponentID(ch);
      DEBUG_STRING_NEW(sDebugBestMode)

      //use RDO to decide whether Cr/Cb takes TS
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[uiFullDepth][CI_QT_TRAFO_ROOT] );

      const Bool splitIntoSubTUs = rTu.getRect(compID).width != rTu.getRect(compID).height;

      TComTURecurse TUIterator(rTu, false, (splitIntoSubTUs ? TComTU::VERTICAL_SPLIT : TComTU::DONT_SPLIT), true, compID);

      const UInt partIdxesPerSubTU = TUIterator.GetAbsPartIdxNumParts(compID);

      do
      {
        const UInt subTUAbsPartIdx   = TUIterator.GetAbsPartIdxTU(compID);

        Double     dSingleCost               = MAX_DOUBLE;
        Int        bestModeId                = 0;
        Distortion singleDistC               = 0;
        UInt       singleCbfC                = 0;
        Distortion singleDistCTmp            = 0;
        Double     singleCostTmp             = 0;
        UInt       singleCbfCTmp             = 0;
        SChar      bestCrossCPredictionAlpha = 0;
        Int        bestTransformSkipMode     = 0;

        const Bool checkCrossComponentPrediction =    (pcCU->getIntraDir(CHANNEL_TYPE_CHROMA, subTUAbsPartIdx) == DM_CHROMA_IDX)
                                                   &&  pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag()
                                                   && (pcCU->getCbf(subTUAbsPartIdx,  COMPONENT_Y, uiTrDepth) != 0);

        const Int  crossCPredictionModesToTest = checkCrossComponentPrediction ? 2 : 1;
        const Int  transformSkipModesToTest    = checkTransformSkip            ? 2 : 1;
        const Int  totalModesToTest            = crossCPredictionModesToTest * transformSkipModesToTest;
              Int  currModeId                  = 0;
              Int  default0Save1Load2          = 0;

        for(Int transformSkipModeId = 0; transformSkipModeId < transformSkipModesToTest; transformSkipModeId++)
        {
          for(Int crossCPredictionModeId = 0; crossCPredictionModeId < crossCPredictionModesToTest; crossCPredictionModeId++)
          {
            pcCU->setCrossComponentPredictionAlphaPartRange(0, compID, subTUAbsPartIdx, partIdxesPerSubTU);
            DEBUG_STRING_NEW(sDebugMode)
            pcCU->setTransformSkipPartRange( transformSkipModeId, compID, subTUAbsPartIdx, partIdxesPerSubTU );
            currModeId++;

            const Bool isOneMode  = (totalModesToTest == 1);
            const Bool isLastMode = (currModeId == totalModesToTest); // currModeId is indexed from 1

            if (isOneMode)
            {
              default0Save1Load2 = 0;
            }
            else if (!isOneMode && (transformSkipModeId == 0) && (crossCPredictionModeId == 0))
            {
              default0Save1Load2 = 1; //save prediction on first mode
            }
            else
            {
              default0Save1Load2 = 2; //load it on subsequent modes
            }

            singleDistCTmp = 0;

            xIntraCodingTUBlock( pcOrgYuv, pcPredYuv, pcResiYuv, resiLuma, (crossCPredictionModeId != 0), singleDistCTmp, compID, TUIterator DEBUG_STRING_PASS_INTO(sDebugMode), default0Save1Load2);
            singleCbfCTmp = pcCU->getCbf( subTUAbsPartIdx, compID, uiTrDepth);

            if (  ((crossCPredictionModeId == 1) && (pcCU->getCrossComponentPredictionAlpha(subTUAbsPartIdx, compID) == 0))
               || ((transformSkipModeId    == 1) && (singleCbfCTmp == 0))) //In order not to code TS flag when cbf is zero, the case for TS with cbf being zero is forbidden.
            {
              singleCostTmp = MAX_DOUBLE;
            }
            else if (!isOneMode)
            {
              UInt bitsTmp = xGetIntraBitsQTChroma( TUIterator, compID, false );
              singleCostTmp  = m_pcRdCost->calcRdCost( bitsTmp, singleDistCTmp);
            }

            if(singleCostTmp < dSingleCost)
            {
              DEBUG_STRING_SWAP(sDebugBestMode, sDebugMode)
              dSingleCost               = singleCostTmp;
              singleDistC               = singleDistCTmp;
              bestCrossCPredictionAlpha = (crossCPredictionModeId != 0) ? pcCU->getCrossComponentPredictionAlpha(subTUAbsPartIdx, compID) : 0;
              bestTransformSkipMode     = transformSkipModeId;
              bestModeId                = currModeId;
              singleCbfC                = singleCbfCTmp;

              if (!isOneMode && !isLastMode)
              {
                xStoreIntraResultQT(compID, TUIterator);
                m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiFullDepth ][ CI_TEMP_BEST ] );
              }
            }

            if (!isOneMode && !isLastMode)
            {
              m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiFullDepth ][ CI_QT_TRAFO_ROOT ] );
            }
          }
        }

        if(bestModeId < totalModesToTest)
        {
          xLoadIntraResultQT(compID, TUIterator);
          pcCU->setCbfPartRange( singleCbfC << uiTrDepth, compID, subTUAbsPartIdx, partIdxesPerSubTU );

          m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiFullDepth ][ CI_TEMP_BEST ] );
        }

        DEBUG_STRING_APPEND(sDebug, sDebugBestMode)
        pcCU ->setTransformSkipPartRange                ( bestTransformSkipMode,     compID, subTUAbsPartIdx, partIdxesPerSubTU );
        pcCU ->setCrossComponentPredictionAlphaPartRange( bestCrossCPredictionAlpha, compID, subTUAbsPartIdx, partIdxesPerSubTU );
        ruiDist += singleDistC;
      } while (TUIterator.nextSection(rTu));

      if (splitIntoSubTUs)
      {
        offsetSubTUCBFs(rTu, compID);
      }
    }
  }
  else
  {
    UInt    uiSplitCbf[MAX_NUM_COMPONENT] = {0,0,0};

    TComTURecurse tuRecurseChild(rTu, false);
    const UInt uiTrDepthChild   = tuRecurseChild.GetTransformDepthRel();
    do
    {
      DEBUG_STRING_NEW(sChild)

      xRecurIntraChromaCodingQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLuma, ruiDist, tuRecurseChild DEBUG_STRING_PASS_INTO(sChild) );

      DEBUG_STRING_APPEND(sDebug, sChild)
      const UInt uiAbsPartIdxSub=tuRecurseChild.GetAbsPartIdxTU();

      for(UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
      {
        uiSplitCbf[ch] |= pcCU->getCbf( uiAbsPartIdxSub, ComponentID(ch), uiTrDepthChild );
      }
    } while ( tuRecurseChild.nextSection(rTu) );


    UInt uiPartsDiv = rTu.GetAbsPartIdxNumParts();
    for(UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
    {
      if (uiSplitCbf[ch])
      {
        const UInt flag=1<<uiTrDepth;
        ComponentID compID=ComponentID(ch);
        UChar *pBase=pcCU->getCbf( compID );
        for( UInt uiOffs = 0; uiOffs < uiPartsDiv; uiOffs++ )
        {
          pBase[ uiAbsPartIdx + uiOffs ] |= flag;
        }
      }
    }
  }
}




Void
TEncSearch::xSetIntraResultChromaQT(TComYuv*    pcRecoYuv, TComTU &rTu)
{
  if (!rTu.ProcessChannelSection(CHANNEL_TYPE_CHROMA))
  {
    return;
  }
  TComDataCU *pcCU=rTu.getCU();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiTrDepth   = rTu.GetTransformDepthRel();
  UInt uiTrMode     = pcCU->getTransformIdx( uiAbsPartIdx );
  if(  uiTrMode == uiTrDepth )
  {
    UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();
    UInt uiQTLayer    = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;

    //===== copy transform coefficients =====
    const TComRectangle &tuRectCb=rTu.getRect(COMPONENT_Cb);
    UInt uiNumCoeffC    = tuRectCb.width*tuRectCb.height;//( pcCU->getSlice()->getSPS()->getMaxCUWidth() * pcCU->getSlice()->getSPS()->getMaxCUHeight() ) >> ( uiFullDepth << 1 );
    const UInt offset = rTu.getCoefficientOffset(COMPONENT_Cb);

    const UInt numberValidComponents = getNumberValidComponents(rTu.GetChromaFormat());
    for (UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
    {
      const ComponentID component = ComponentID(ch);
      const TCoeff* src           = m_ppcQTTempCoeff[component][uiQTLayer] + offset;//(uiNumCoeffIncC*uiAbsPartIdx);
      TCoeff* dest                = pcCU->getCoeff(component) + offset;//(uiNumCoeffIncC*uiAbsPartIdx);
      ::memcpy( dest, src, sizeof(TCoeff)*uiNumCoeffC );
#if ADAPTIVE_QP_SELECTION
      TCoeff* pcArlCoeffSrc = m_ppcQTTempArlCoeff[component][ uiQTLayer ] + offset;//( uiNumCoeffIncC * uiAbsPartIdx );
      TCoeff* pcArlCoeffDst = pcCU->getArlCoeff(component)                + offset;//( uiNumCoeffIncC * uiAbsPartIdx );
      ::memcpy( pcArlCoeffDst, pcArlCoeffSrc, sizeof( TCoeff ) * uiNumCoeffC );
#endif
    }

    //===== copy reconstruction =====

    m_pcQTTempTComYuv[ uiQTLayer ].copyPartToPartComponent( COMPONENT_Cb, pcRecoYuv, uiAbsPartIdx, tuRectCb.width, tuRectCb.height );
    m_pcQTTempTComYuv[ uiQTLayer ].copyPartToPartComponent( COMPONENT_Cr, pcRecoYuv, uiAbsPartIdx, tuRectCb.width, tuRectCb.height );
  }
  else
  {
    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xSetIntraResultChromaQT( pcRecoYuv, tuRecurseChild );
    } while (tuRecurseChild.nextSection(rTu));
  }
}



Void
TEncSearch::estIntraPredLumaQT(TComDataCU* pcCU,
                               TComYuv*    pcOrgYuv,
                               TComYuv*    pcPredYuv,
                               TComYuv*    pcResiYuv,
                               TComYuv*    pcRecoYuv,
                               Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                               DEBUG_STRING_FN_DECLARE(sDebug))
{
  const UInt         uiDepth               = pcCU->getDepth(0);
  const UInt         uiInitTrDepth         = pcCU->getPartitionSize(0) == SIZE_2Nx2N ? 0 : 1;
  const UInt         uiNumPU               = 1<<(2*uiInitTrDepth);
  const UInt         uiQNumParts           = pcCU->getTotalNumPart() >> 2;
  const UInt         uiWidthBit            = pcCU->getIntraSizeIdx(0);
  const ChromaFormat chFmt                 = pcCU->getPic()->getChromaFormat();
  const UInt         numberValidComponents = getNumberValidComponents(chFmt);
  const TComSPS     &sps                   = *(pcCU->getSlice()->getSPS());
  const TComPPS     &pps                   = *(pcCU->getSlice()->getPPS());
        Distortion   uiOverallDistY        = 0;
        UInt         CandNum;
        Double       CandCostList[ FAST_UDI_MAX_RDMODE_NUM ];
        Pel          resiLumaPU[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];

        Bool    bMaintainResidual[NUMBER_OF_STORED_RESIDUAL_TYPES];
        for (UInt residualTypeIndex = 0; residualTypeIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; residualTypeIndex++)
        {
          bMaintainResidual[residualTypeIndex] = true; //assume true unless specified otherwise
        }

        bMaintainResidual[RESIDUAL_ENCODER_SIDE] = !(m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate());

  // Lambda calculation at equivalent Qp of 4 is recommended because at that Qp, the quantisation divisor is 1.
#if FULL_NBIT
  const Double sqrtLambdaForFirstPass= (m_pcEncCfg->getCostMode()==COST_MIXED_LOSSLESS_LOSSY_CODING && pcCU->getCUTransquantBypass(0)) ?
                sqrt(0.57 * pow(2.0, ((LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME - 12) / 3.0)))
              : m_pcRdCost->getSqrtLambda();
#else
  const Double sqrtLambdaForFirstPass= (m_pcEncCfg->getCostMode()==COST_MIXED_LOSSLESS_LOSSY_CODING && pcCU->getCUTransquantBypass(0)) ?
                sqrt(0.57 * pow(2.0, ((LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME - 12 - 6 * (sps.getBitDepth(CHANNEL_TYPE_LUMA) - 8)) / 3.0)))
              : m_pcRdCost->getSqrtLambda();
#endif

  //===== set QP and clear Cbf =====
  if ( pps.getUseDQP() == true)
  {
    pcCU->setQPSubParts( pcCU->getQP(0), 0, uiDepth );
  }
  else
  {
    pcCU->setQPSubParts( pcCU->getSlice()->getSliceQp(), 0, uiDepth );
  }

  //===== loop over partitions =====
  TComTURecurse tuRecurseCU(pcCU, 0);
  TComTURecurse tuRecurseWithPU(tuRecurseCU, false, (uiInitTrDepth==0)?TComTU::DONT_SPLIT : TComTU::QUAD_SPLIT);

  do
  {
    const UInt uiPartOffset=tuRecurseWithPU.GetAbsPartIdxTU();
//  for( UInt uiPU = 0, uiPartOffset=0; uiPU < uiNumPU; uiPU++, uiPartOffset += uiQNumParts )
  //{
    //===== init pattern for luma prediction =====
    DEBUG_STRING_NEW(sTemp2)

    //===== determine set of modes to be tested (using prediction signal only) =====
    Int numModesAvailable     = 35; //total number of Intra modes
    UInt uiRdModeList[FAST_UDI_MAX_RDMODE_NUM];
    Int numModesForFullRD = m_pcEncCfg->getFastUDIUseMPMEnabled()?g_aucIntraModeNumFast_UseMPM[ uiWidthBit ] : g_aucIntraModeNumFast_NotUseMPM[ uiWidthBit ];

    // this should always be true
    assert (tuRecurseWithPU.ProcessComponentSection(COMPONENT_Y));
    initIntraPatternChType( tuRecurseWithPU, COMPONENT_Y, true DEBUG_STRING_PASS_INTO(sTemp2) );

    Bool doFastSearch = (numModesForFullRD != numModesAvailable);
    if (doFastSearch)
    {
      assert(numModesForFullRD < numModesAvailable);

      for( Int i=0; i < numModesForFullRD; i++ )
      {
        CandCostList[ i ] = MAX_DOUBLE;
      }
      CandNum = 0;

      const TComRectangle &puRect=tuRecurseWithPU.getRect(COMPONENT_Y);
      const UInt uiAbsPartIdx=tuRecurseWithPU.GetAbsPartIdxTU();

      Pel* piOrg         = pcOrgYuv ->getAddr( COMPONENT_Y, uiAbsPartIdx );
      Pel* piPred        = pcPredYuv->getAddr( COMPONENT_Y, uiAbsPartIdx );
      UInt uiStride      = pcPredYuv->getStride( COMPONENT_Y );
      DistParam distParam;
      const Bool bUseHadamard=pcCU->getCUTransquantBypass(0) == 0;
      m_pcRdCost->setDistParam(distParam, sps.getBitDepth(CHANNEL_TYPE_LUMA), piOrg, uiStride, piPred, uiStride, puRect.width, puRect.height, bUseHadamard);
      distParam.bApplyWeight = false;
      for( Int modeIdx = 0; modeIdx < numModesAvailable; modeIdx++ )
      {
        UInt       uiMode = modeIdx;
        Distortion uiSad  = 0;

        const Bool bUseFilter=TComPrediction::filteringIntraReferenceSamples(COMPONENT_Y, uiMode, puRect.width, puRect.height, chFmt, sps.getSpsRangeExtension().getIntraSmoothingDisabledFlag());

        predIntraAng( COMPONENT_Y, uiMode, piOrg, uiStride, piPred, uiStride, tuRecurseWithPU, bUseFilter, TComPrediction::UseDPCMForFirstPassIntraEstimation(tuRecurseWithPU, uiMode) );

        // use hadamard transform here
        uiSad+=distParam.DistFunc(&distParam);

        UInt   iModeBits = 0;

        // NB xModeBitsIntra will not affect the mode for chroma that may have already been pre-estimated.
        iModeBits+=xModeBitsIntra( pcCU, uiMode, uiPartOffset, uiDepth, CHANNEL_TYPE_LUMA );

        Double cost      = (Double)uiSad + (Double)iModeBits * sqrtLambdaForFirstPass;

#if DEBUG_INTRA_SEARCH_COSTS
        std::cout << "1st pass mode " << uiMode << " SAD = " << uiSad << ", mode bits = " << iModeBits << ", cost = " << cost << "\n";
#endif

        CandNum += xUpdateCandList( uiMode, cost, numModesForFullRD, uiRdModeList, CandCostList );
      }

      if (m_pcEncCfg->getFastUDIUseMPMEnabled())
      {
        Int uiPreds[NUM_MOST_PROBABLE_MODES] = {-1, -1, -1};

        Int iMode = -1;
        pcCU->getIntraDirPredictor( uiPartOffset, uiPreds, COMPONENT_Y, &iMode );

        const Int numCand = ( iMode >= 0 ) ? iMode : Int(NUM_MOST_PROBABLE_MODES);

        for( Int j=0; j < numCand; j++)
        {
          Bool mostProbableModeIncluded = false;
          Int mostProbableMode = uiPreds[j];

          for( Int i=0; i < numModesForFullRD; i++)
          {
            mostProbableModeIncluded |= (mostProbableMode == uiRdModeList[i]);
          }
          if (!mostProbableModeIncluded)
          {
            uiRdModeList[numModesForFullRD++] = mostProbableMode;
          }
        }
      }
    }
    else
    {
      for( Int i=0; i < numModesForFullRD; i++)
      {
        uiRdModeList[i] = i;
      }
    }

    //===== check modes (using r-d costs) =====
#if HHI_RQT_INTRA_SPEEDUP_MOD
    UInt   uiSecondBestMode  = MAX_UINT;
    Double dSecondBestPUCost = MAX_DOUBLE;
#endif
    DEBUG_STRING_NEW(sPU)
    UInt       uiBestPUMode  = 0;
    Distortion uiBestPUDistY = 0;
    Double     dBestPUCost   = MAX_DOUBLE;

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
    UInt max=numModesForFullRD;

    if (DebugOptionList::ForceLumaMode.isSet())
    {
      max=0;  // we are forcing a direction, so don't bother with mode check
    }
    for ( UInt uiMode = 0; uiMode < max; uiMode++)
#else
    for( UInt uiMode = 0; uiMode < numModesForFullRD; uiMode++ )
#endif
    {
      // set luma prediction mode
      UInt uiOrgMode = uiRdModeList[uiMode];

      pcCU->setIntraDirSubParts ( CHANNEL_TYPE_LUMA, uiOrgMode, uiPartOffset, uiDepth + uiInitTrDepth );

      DEBUG_STRING_NEW(sMode)
      // set context models
      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST] );

      // determine residual for partition
      Distortion uiPUDistY = 0;
      Double     dPUCost   = 0.0;
#if HHI_RQT_INTRA_SPEEDUP
      xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaPU, uiPUDistY, true, dPUCost, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sMode) );
#else
      xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaPU, uiPUDistY, dPUCost, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sMode) );
#endif

#if DEBUG_INTRA_SEARCH_COSTS
      std::cout << "2nd pass [luma,chroma] mode [" << Int(pcCU->getIntraDir(CHANNEL_TYPE_LUMA, uiPartOffset)) << "," << Int(pcCU->getIntraDir(CHANNEL_TYPE_CHROMA, uiPartOffset)) << "] cost = " << dPUCost << "\n";
#endif

      // check r-d cost
      if( dPUCost < dBestPUCost )
      {
        DEBUG_STRING_SWAP(sPU, sMode)
#if HHI_RQT_INTRA_SPEEDUP_MOD
        uiSecondBestMode  = uiBestPUMode;
        dSecondBestPUCost = dBestPUCost;
#endif
        uiBestPUMode  = uiOrgMode;
        uiBestPUDistY = uiPUDistY;
        dBestPUCost   = dPUCost;

        xSetIntraResultLumaQT( pcRecoYuv, tuRecurseWithPU );

        if (pps.getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
        {
          const Int xOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).x0;
          const Int yOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).y0;
          for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
          {
            if (bMaintainResidual[storedResidualIndex])
            {
              xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaPU[storedResidualIndex], tuRecurseWithPU, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE );
            }
          }
        }

        UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();

        ::memcpy( m_puhQTTempTrIdx,  pcCU->getTransformIdx()       + uiPartOffset, uiQPartNum * sizeof( UChar ) );
        for (UInt component = 0; component < numberValidComponents; component++)
        {
          const ComponentID compID = ComponentID(component);
          ::memcpy( m_puhQTTempCbf[compID], pcCU->getCbf( compID  ) + uiPartOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_puhQTTempTransformSkipFlag[compID],  pcCU->getTransformSkip(compID)  + uiPartOffset, uiQPartNum * sizeof( UChar ) );
        }
      }
#if HHI_RQT_INTRA_SPEEDUP_MOD
      else if( dPUCost < dSecondBestPUCost )
      {
        uiSecondBestMode  = uiOrgMode;
        dSecondBestPUCost = dPUCost;
      }
#endif
    } // Mode loop

#if HHI_RQT_INTRA_SPEEDUP
#if HHI_RQT_INTRA_SPEEDUP_MOD
    for( UInt ui =0; ui < 2; ++ui )
#endif
    if(!m_pcEncCfg->getTransquantBypassInferTUSplit() || !pcCU->isLosslessCoded(0))
    {
#if HHI_RQT_INTRA_SPEEDUP_MOD
      UInt uiOrgMode   = ui ? uiSecondBestMode  : uiBestPUMode;
      if( uiOrgMode == MAX_UINT )
      {
        break;
      }
#else
      UInt uiOrgMode = uiBestPUMode;
#endif

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
      if (DebugOptionList::ForceLumaMode.isSet())
      {
        uiOrgMode = DebugOptionList::ForceLumaMode.getInt();
      }
#endif

      pcCU->setIntraDirSubParts ( CHANNEL_TYPE_LUMA, uiOrgMode, uiPartOffset, uiDepth + uiInitTrDepth );
      DEBUG_STRING_NEW(sModeTree)

      // set context models
      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST] );

      // determine residual for partition
      Distortion uiPUDistY = 0;
      Double     dPUCost   = 0.0;

      xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaPU, uiPUDistY, false, dPUCost, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sModeTree));

      // check r-d cost
      if( dPUCost < dBestPUCost )
      {
        DEBUG_STRING_SWAP(sPU, sModeTree)
        uiBestPUMode  = uiOrgMode;
        uiBestPUDistY = uiPUDistY;
        dBestPUCost   = dPUCost;

        xSetIntraResultLumaQT( pcRecoYuv, tuRecurseWithPU );

        if (pps.getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
        {
          const Int xOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).x0;
          const Int yOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).y0;
          for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
          {
            if (bMaintainResidual[storedResidualIndex])
            {
              xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaPU[storedResidualIndex], tuRecurseWithPU, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE );
            }
          }
        }

        const UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();
        ::memcpy( m_puhQTTempTrIdx,  pcCU->getTransformIdx()       + uiPartOffset, uiQPartNum * sizeof( UChar ) );

        for (UInt component = 0; component < numberValidComponents; component++)
        {
          const ComponentID compID = ComponentID(component);
          ::memcpy( m_puhQTTempCbf[compID], pcCU->getCbf( compID  ) + uiPartOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_puhQTTempTransformSkipFlag[compID],  pcCU->getTransformSkip(compID)  + uiPartOffset, uiQPartNum * sizeof( UChar ) );
        }
      }
    } // Mode loop
#endif

    DEBUG_STRING_APPEND(sDebug, sPU)

    //--- update overall distortion ---
    uiOverallDistY += uiBestPUDistY;

    //--- update transform index and cbf ---
    const UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();
    ::memcpy( pcCU->getTransformIdx()       + uiPartOffset, m_puhQTTempTrIdx,  uiQPartNum * sizeof( UChar ) );
    for (UInt component = 0; component < numberValidComponents; component++)
    {
      const ComponentID compID = ComponentID(component);
      ::memcpy( pcCU->getCbf( compID  ) + uiPartOffset, m_puhQTTempCbf[compID], uiQPartNum * sizeof( UChar ) );
      ::memcpy( pcCU->getTransformSkip( compID  ) + uiPartOffset, m_puhQTTempTransformSkipFlag[compID ], uiQPartNum * sizeof( UChar ) );
    }

    //--- set reconstruction for next intra prediction blocks ---
    if( !tuRecurseWithPU.IsLastSection() )
    {
      const TComRectangle &puRect=tuRecurseWithPU.getRect(COMPONENT_Y);
      const UInt  uiCompWidth   = puRect.width;
      const UInt  uiCompHeight  = puRect.height;

      const UInt  uiZOrder      = pcCU->getZorderIdxInCtu() + uiPartOffset;
            Pel*  piDes         = pcCU->getPic()->getPicYuvRec()->getAddr( COMPONENT_Y, pcCU->getCtuRsAddr(), uiZOrder );
      const UInt  uiDesStride   = pcCU->getPic()->getPicYuvRec()->getStride( COMPONENT_Y);
      const Pel*  piSrc         = pcRecoYuv->getAddr( COMPONENT_Y, uiPartOffset );
      const UInt  uiSrcStride   = pcRecoYuv->getStride( COMPONENT_Y);

      for( UInt uiY = 0; uiY < uiCompHeight; uiY++, piSrc += uiSrcStride, piDes += uiDesStride )
      {
        for( UInt uiX = 0; uiX < uiCompWidth; uiX++ )
        {
          piDes[ uiX ] = piSrc[ uiX ];
        }
      }
    }

    //=== update PU data ====
    pcCU->setIntraDirSubParts     ( CHANNEL_TYPE_LUMA, uiBestPUMode, uiPartOffset, uiDepth + uiInitTrDepth );
  } while (tuRecurseWithPU.nextSection(tuRecurseCU));


  if( uiNumPU > 1 )
  { // set Cbf for all blocks
    UInt uiCombCbfY = 0;
    UInt uiCombCbfU = 0;
    UInt uiCombCbfV = 0;
    UInt uiPartIdx  = 0;
    for( UInt uiPart = 0; uiPart < 4; uiPart++, uiPartIdx += uiQNumParts )
    {
      uiCombCbfY |= pcCU->getCbf( uiPartIdx, COMPONENT_Y,  1 );
      uiCombCbfU |= pcCU->getCbf( uiPartIdx, COMPONENT_Cb, 1 );
      uiCombCbfV |= pcCU->getCbf( uiPartIdx, COMPONENT_Cr, 1 );
    }
    for( UInt uiOffs = 0; uiOffs < 4 * uiQNumParts; uiOffs++ )
    {
      pcCU->getCbf( COMPONENT_Y  )[ uiOffs ] |= uiCombCbfY;
      pcCU->getCbf( COMPONENT_Cb )[ uiOffs ] |= uiCombCbfU;
      pcCU->getCbf( COMPONENT_Cr )[ uiOffs ] |= uiCombCbfV;
    }
  }

  //===== reset context models =====
  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);

  //===== set distortion (rate and r-d costs are determined later) =====
  pcCU->getTotalDistortion() = uiOverallDistY;
}




Void
TEncSearch::estIntraPredChromaQT(TComDataCU* pcCU,
                                 TComYuv*    pcOrgYuv,
                                 TComYuv*    pcPredYuv,
                                 TComYuv*    pcResiYuv,
                                 TComYuv*    pcRecoYuv,
                                 Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                 DEBUG_STRING_FN_DECLARE(sDebug))
{
  assert( pcCU->getColourTransform( 0 ) == false );

  const UInt    uiInitTrDepth  = pcCU->getPartitionSize(0) != SIZE_2Nx2N && enable4ChromaPUsInIntraNxNCU(pcOrgYuv->getChromaFormat()) ? 1 : 0;

  TComTURecurse tuRecurseCU(pcCU, 0);
  TComTURecurse tuRecurseWithPU(tuRecurseCU, false, (uiInitTrDepth==0)?TComTU::DONT_SPLIT : TComTU::QUAD_SPLIT);
  const UInt    uiQNumParts    = tuRecurseWithPU.GetAbsPartIdxNumParts();
  const UInt    uiDepthCU=tuRecurseWithPU.getCUDepth();
  const UInt    numberValidComponents = pcCU->getPic()->getNumberValidComponents();

  do
  {
    UInt       uiBestMode  = 0;
    Distortion uiBestDist  = 0;
    Double     dBestCost   = MAX_DOUBLE;

    //----- init mode list -----
    if (tuRecurseWithPU.ProcessChannelSection(CHANNEL_TYPE_CHROMA))
    {
      UInt uiModeList[FAST_UDI_MAX_RDMODE_NUM];
      const UInt  uiQPartNum     = uiQNumParts;
      const UInt  uiPartOffset   = tuRecurseWithPU.GetAbsPartIdxTU();
      {
        UInt  uiMinMode = 0;
        UInt  uiMaxMode = NUM_CHROMA_MODE;

        //----- check chroma modes -----
        pcCU->getAllowedChromaDir( uiPartOffset, uiModeList );

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
        if (DebugOptionList::ForceChromaMode.isSet())
        {
          uiMinMode=DebugOptionList::ForceChromaMode.getInt();
          if (uiModeList[uiMinMode]==34)
          {
            uiMinMode=4; // if the fixed mode has been renumbered because DM_CHROMA covers it, use DM_CHROMA.
          }
          uiMaxMode=uiMinMode+1;
        }
#endif

        DEBUG_STRING_NEW(sPU)

        for( UInt uiMode = uiMinMode; uiMode < uiMaxMode; uiMode++ )
        {
          //----- restore context models -----
          m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[uiDepthCU][CI_CURR_BEST] );
          
          DEBUG_STRING_NEW(sMode)
          //----- chroma coding -----
          Distortion uiDist = 0;
          pcCU->setIntraDirSubParts  ( CHANNEL_TYPE_CHROMA, uiModeList[uiMode], uiPartOffset, uiDepthCU+uiInitTrDepth );
          xRecurIntraChromaCodingQT       ( pcOrgYuv, pcPredYuv, pcResiYuv, resiLuma, uiDist, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sMode) );

          if( pcCU->getSlice()->getPPS()->getUseTransformSkip() )
          {
            m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[uiDepthCU][CI_CURR_BEST] );
          }

          UInt    uiBits = xGetIntraBitsQT( tuRecurseWithPU, false, true, false );
          Double  dCost  = m_pcRdCost->calcRdCost( uiBits, uiDist );

          //----- compare -----
          if( dCost < dBestCost )
          {
            DEBUG_STRING_SWAP(sPU, sMode);
            dBestCost   = dCost;
            uiBestDist  = uiDist;
            uiBestMode  = uiModeList[uiMode];

            xSetIntraResultChromaQT( pcRecoYuv, tuRecurseWithPU );
            for (UInt componentIndex = COMPONENT_Cb; componentIndex < numberValidComponents; componentIndex++)
            {
              const ComponentID compID = ComponentID(componentIndex);
              ::memcpy( m_puhQTTempCbf[compID], pcCU->getCbf( compID )+uiPartOffset, uiQPartNum * sizeof( UChar ) );
              ::memcpy( m_puhQTTempTransformSkipFlag[compID], pcCU->getTransformSkip( compID )+uiPartOffset, uiQPartNum * sizeof( UChar ) );
              ::memcpy( m_phQTTempCrossComponentPredictionAlpha[compID], pcCU->getCrossComponentPredictionAlpha(compID)+uiPartOffset, uiQPartNum * sizeof( SChar ) );
            }
          }
        }

        DEBUG_STRING_APPEND(sDebug, sPU)

        //----- set data -----
        for (UInt componentIndex = COMPONENT_Cb; componentIndex < numberValidComponents; componentIndex++)
        {
          const ComponentID compID = ComponentID(componentIndex);
          ::memcpy( pcCU->getCbf( compID )+uiPartOffset, m_puhQTTempCbf[compID], uiQPartNum * sizeof( UChar ) );
          ::memcpy( pcCU->getTransformSkip( compID )+uiPartOffset, m_puhQTTempTransformSkipFlag[compID], uiQPartNum * sizeof( UChar ) );
          ::memcpy( pcCU->getCrossComponentPredictionAlpha(compID)+uiPartOffset, m_phQTTempCrossComponentPredictionAlpha[compID], uiQPartNum * sizeof( SChar ) );
        }
      }

      if( ! tuRecurseWithPU.IsLastSection() )
      {
        for (UInt ch=COMPONENT_Cb; ch<numberValidComponents; ch++)
        {
          const ComponentID compID    = ComponentID(ch);
          const TComRectangle &tuRect = tuRecurseWithPU.getRect(compID);
          const UInt  uiCompWidth     = tuRect.width;
          const UInt  uiCompHeight    = tuRect.height;
          const UInt  uiZOrder        = pcCU->getZorderIdxInCtu() + tuRecurseWithPU.GetAbsPartIdxTU();
                Pel*  piDes           = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), uiZOrder );
          const UInt  uiDesStride     = pcCU->getPic()->getPicYuvRec()->getStride( compID);
          const Pel*  piSrc           = pcRecoYuv->getAddr( compID, uiPartOffset );
          const UInt  uiSrcStride     = pcRecoYuv->getStride( compID);

          for( UInt uiY = 0; uiY < uiCompHeight; uiY++, piSrc += uiSrcStride, piDes += uiDesStride )
          {
            for( UInt uiX = 0; uiX < uiCompWidth; uiX++ )
            {
              piDes[ uiX ] = piSrc[ uiX ];
            }
          }
        }
      }

      pcCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, uiBestMode, uiPartOffset, uiDepthCU+uiInitTrDepth );
      pcCU->getTotalDistortion      () += uiBestDist;
    }

  } while (tuRecurseWithPU.nextSection(tuRecurseCU));

  //----- restore context models -----

  if( uiInitTrDepth != 0 )
  { // set Cbf for all blocks
    UInt uiCombCbfU = 0;
    UInt uiCombCbfV = 0;
    UInt uiPartIdx  = 0;
    for( UInt uiPart = 0; uiPart < 4; uiPart++, uiPartIdx += uiQNumParts )
    {
      uiCombCbfU |= pcCU->getCbf( uiPartIdx, COMPONENT_Cb, 1 );
      uiCombCbfV |= pcCU->getCbf( uiPartIdx, COMPONENT_Cr, 1 );
    }
    for( UInt uiOffs = 0; uiOffs < 4 * uiQNumParts; uiOffs++ )
    {
      pcCU->getCbf( COMPONENT_Cb )[ uiOffs ] |= uiCombCbfU;
      pcCU->getCbf( COMPONENT_Cr )[ uiOffs ] |= uiCombCbfV;
    }
  }

  m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[uiDepthCU][CI_CURR_BEST] );
}




/** Function for encoding and reconstructing luma/chroma samples of a PCM mode CU.
 * \param pcCU pointer to current CU
 * \param uiAbsPartIdx part index
 * \param pOrg pointer to original sample arrays
 * \param pPCM pointer to PCM code arrays
 * \param pPred pointer to prediction signal arrays
 * \param pResi pointer to residual signal arrays
 * \param pReco pointer to reconstructed sample arrays
 * \param uiStride stride of the original/prediction/residual sample arrays
 * \param uiWidth block width
 * \param uiHeight block height
 * \param compID texture component type
 */
Void TEncSearch::xEncPCM (TComDataCU* pcCU, UInt uiAbsPartIdx, Pel* pOrg, Pel* pPCM, Pel* pPred, Pel* pResi, Pel* pReco, UInt uiStride, UInt uiWidth, UInt uiHeight, const ComponentID compID )
{
  const UInt uiReconStride   = pcCU->getPic()->getPicYuvRec()->getStride(compID);
  const UInt uiPCMBitDepth   = pcCU->getSlice()->getSPS()->getPCMBitDepth(toChannelType(compID));
  const Int  channelBitDepth = pcCU->getSlice()->getSPS()->getBitDepth(toChannelType(compID));
  Pel* pRecoPic = pcCU->getPic()->getPicYuvRec()->getAddr(compID, pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu()+uiAbsPartIdx);

  const Int pcmShiftRight=(channelBitDepth - Int(uiPCMBitDepth));

  assert(pcmShiftRight >= 0);

  for( UInt uiY = 0; uiY < uiHeight; uiY++ )
  {
    for( UInt uiX = 0; uiX < uiWidth; uiX++ )
    {
      // Reset pred and residual
      pPred[uiX] = 0;
      pResi[uiX] = 0;
      // Encode
      pPCM[uiX] = (pOrg[uiX]>>pcmShiftRight);
      // Reconstruction
      pReco   [uiX] = (pPCM[uiX]<<(pcmShiftRight));
      pRecoPic[uiX] = pReco[uiX];
    }
    pPred += uiStride;
    pResi += uiStride;
    pPCM += uiWidth;
    pOrg += uiStride;
    pReco += uiStride;
    pRecoPic += uiReconStride;
  }
}


//!  Function for PCM mode estimation.
Void TEncSearch::IPCMSearch( TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcPredYuv, TComYuv* pcResiYuv, TComYuv* pcRecoYuv )
{
  UInt              uiDepth      = pcCU->getDepth(0);
  const Distortion  uiDistortion = 0;
  UInt              uiBits;

  Double dCost;

  for (UInt ch=0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
  {
    const ComponentID compID  = ComponentID(ch);
    const UInt width  = pcCU->getWidth(0)  >> pcCU->getPic()->getComponentScaleX(compID);
    const UInt height = pcCU->getHeight(0) >> pcCU->getPic()->getComponentScaleY(compID);
    const UInt stride = pcPredYuv->getStride(compID);

    Pel * pOrig    = pcOrgYuv->getAddr  (compID, 0, width);
    Pel * pResi    = pcResiYuv->getAddr(compID, 0, width);
    Pel * pPred    = pcPredYuv->getAddr(compID, 0, width);
    Pel * pReco    = pcRecoYuv->getAddr(compID, 0, width);
    Pel * pPCM     = pcCU->getPCMSample (compID);

    xEncPCM ( pcCU, 0, pOrig, pPCM, pPred, pResi, pReco, stride, width, height, compID );

  }

  m_pcEntropyCoder->resetBits();
  xEncIntraHeader ( pcCU, uiDepth, 0, true, false);
  uiBits = m_pcEntropyCoder->getNumberOfWrittenBits();

  dCost = m_pcRdCost->calcRdCost( uiBits, uiDistortion );

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);

  pcCU->getTotalBits()       = uiBits;
  pcCU->getTotalCost()       = dCost;
  pcCU->getTotalDistortion() = uiDistortion;

  pcCU->copyToPic(uiDepth);
}




Void TEncSearch::xGetInterPredictionError( TComDataCU* pcCU, TComYuv* pcYuvOrg, Int iPartIdx, Distortion& ruiErr, Bool /*bHadamard*/ )
{
  motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, iPartIdx );

  UInt uiAbsPartIdx = 0;
  Int iWidth = 0;
  Int iHeight = 0;
  pcCU->getPartIndexAndSize( iPartIdx, uiAbsPartIdx, iWidth, iHeight );

  DistParam cDistParam;

  cDistParam.bApplyWeight = false;


  m_pcRdCost->setDistParam( cDistParam, pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA),
                            pcYuvOrg->getAddr( COMPONENT_Y, uiAbsPartIdx ), pcYuvOrg->getStride(COMPONENT_Y),
                            m_tmpYuvPred .getAddr( COMPONENT_Y, uiAbsPartIdx ), m_tmpYuvPred.getStride(COMPONENT_Y),
                            iWidth, iHeight, m_pcEncCfg->getUseHADME() && (pcCU->getCUTransquantBypass(uiAbsPartIdx) == 0) );

  ruiErr = cDistParam.DistFunc( &cDistParam );
}

//! estimation of best merge coding
Void TEncSearch::xMergeEstimation( TComDataCU* pcCU, TComYuv* pcYuvOrg, Int iPUIdx, UInt& uiInterDir, TComMvField* pacMvField, UInt& uiMergeIndex, Distortion& ruiCost, TComMvField* cMvFieldNeighbours, UChar* uhInterDirNeighbours, Int& numValidMergeCand, Int iCostCalcType )
{
  UInt uiAbsPartIdx = 0;
  Int iWidth = 0;
  Int iHeight = 0;

  pcCU->getPartIndexAndSize( iPUIdx, uiAbsPartIdx, iWidth, iHeight );
  UInt uiDepth = pcCU->getDepth( uiAbsPartIdx );

  PartSize partSize = pcCU->getPartitionSize( 0 );
  if ( pcCU->getSlice()->getPPS()->getLog2ParallelMergeLevelMinus2() && partSize != SIZE_2Nx2N && pcCU->getWidth( 0 ) <= 8 )
  {
    if ( iPUIdx == 0 )
    {
      pcCU->setPartSizeSubParts( SIZE_2Nx2N, 0, uiDepth ); // temporarily set
#if MCTS_ENC_CHECK
      UInt numSpatialMergeCandidates = 0;
      pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
      if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
      {
        numValidMergeCand = numSpatialMergeCandidates;
      }
#else
      pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours,uhInterDirNeighbours, numValidMergeCand );
#endif
      pcCU->setPartSizeSubParts( partSize, 0, uiDepth ); // restore
    }
  }
  else
  {
#if MCTS_ENC_CHECK
    UInt numSpatialMergeCandidates = 0;
    pcCU->getInterMergeCandidates( uiAbsPartIdx, iPUIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
    {
      numValidMergeCand = numSpatialMergeCandidates;
    }
#else
    pcCU->getInterMergeCandidates( uiAbsPartIdx, iPUIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );
#endif
  }

  pcCU->roundMergeCandidates(cMvFieldNeighbours, numValidMergeCand);
  xRestrictBipredMergeCand( pcCU, iPUIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );

  ruiCost = std::numeric_limits<Distortion>::max();
  if ( iCostCalcType )
  {
    m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( uiAbsPartIdx ) );
  }
  
  for( UInt uiMergeCand = 0; uiMergeCand < numValidMergeCand; ++uiMergeCand )
  {
    if ( (uhInterDirNeighbours[uiMergeCand] == 1 || uhInterDirNeighbours[uiMergeCand] == 3) && pcCU->getSlice()->getRefPic( REF_PIC_LIST_0, cMvFieldNeighbours[uiMergeCand<<1].getRefIdx() )->getPOC() == pcCU->getSlice()->getPOC() )
    {
      continue;
    }
    Distortion uiCostCand = std::numeric_limits<Distortion>::max();
    UInt       uiBitsCand = 0;

    PartSize ePartSize = pcCU->getPartitionSize( 0 );

    pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvField( cMvFieldNeighbours[0 + 2*uiMergeCand], ePartSize, uiAbsPartIdx, 0, iPUIdx );
    pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvField( cMvFieldNeighbours[1 + 2*uiMergeCand], ePartSize, uiAbsPartIdx, 0, iPUIdx );

    if ( !iCostCalcType )
    {
      xGetInterPredictionError( pcCU, pcYuvOrg, iPUIdx, uiCostCand, m_pcEncCfg->getUseHADME() );
    }
    else
    {
      motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, iPUIdx );

      uiCostCand = 0;
      for (Int ch = COMPONENT_Y; ch < (iCostCalcType==1?pcCU->getPic()->getNumberValidComponents(): COMPONENT_Y+1); ch++)
      {
        Int iTempWidth = iWidth >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));
        Int iTempHeight = iHeight >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));
        Int iRefStride = m_tmpYuvPred.getStride(ComponentID(ch));
        Int iOrgStride = pcYuvOrg->getStride(ComponentID(ch));
        Pel *pRef =  m_tmpYuvPred.getAddr( ComponentID(ch), uiAbsPartIdx);
        Pel *pOrg = pcYuvOrg->getAddr( ComponentID(ch), uiAbsPartIdx);

        uiCostCand += getSAD( pRef, iRefStride, pOrg, iOrgStride, iTempWidth, iTempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );

        if ( uiCostCand >= ruiCost )
        {
          break;
        }
      }
    }

    uiBitsCand = uiMergeCand + 1;
    if (uiMergeCand == m_pcEncCfg->getMaxNumMergeCand() -1)
    {
        uiBitsCand--;
    }
    uiCostCand = uiCostCand + m_pcRdCost->getCost( uiBitsCand );
    if ( uiCostCand < ruiCost )
    {
      ruiCost = uiCostCand;
      pacMvField[0] = cMvFieldNeighbours[0 + 2*uiMergeCand];
      pacMvField[1] = cMvFieldNeighbours[1 + 2*uiMergeCand];
      uiInterDir = uhInterDirNeighbours[uiMergeCand];
      uiMergeIndex = uiMergeCand;
    }
  }
}

/** convert bi-pred merge candidates to uni-pred
 * \param pcCU
 * \param puIdx
 * \param mvFieldNeighbours
 * \param interDirNeighbours
 * \param numValidMergeCand
 * \returns Void
 */
Void TEncSearch::xRestrictBipredMergeCand( TComDataCU* pcCU, UInt puIdx, TComMvField* mvFieldNeighbours, UChar* interDirNeighbours, Int numValidMergeCand )
{
  {
    for( UInt mergeCand = 0; mergeCand < numValidMergeCand; ++mergeCand )
    {
      if ( interDirNeighbours[mergeCand] == 3 )
      {
      Bool b8x8BiPredRestricted = pcCU->is8x8BipredRestriction(
        mvFieldNeighbours[(mergeCand << 1)].getMv(),
        mvFieldNeighbours[(mergeCand << 1) + 1].getMv(),
        mvFieldNeighbours[(mergeCand << 1)].getRefIdx(),
        mvFieldNeighbours[(mergeCand << 1) + 1].getRefIdx()
        );
 
      if (pcCU->isBipredRestriction(puIdx,b8x8BiPredRestricted))
      {
        interDirNeighbours[mergeCand] = 1;
        mvFieldNeighbours[(mergeCand << 1) + 1].setMvField(TComMv(0,0), -1);
      }
     }
    }
  }
}

//! search of the best candidate for inter prediction
#if AMP_MRG
Bool TEncSearch::predInterSearch( TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcPredYuv, TComYuv* pcResiYuv, TComYuv* pcRecoYuv DEBUG_STRING_FN_DECLARE(sDebug), Bool bUseRes, Bool bUseMRG, TComMv* iMVCandList )
#else
Bool TEncSearch::predInterSearch( TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcPredYuv, TComYuv* pcResiYuv, TComYuv* pcRecoYuv, Bool bUseRes, TComMv* iMVCandList )
#endif
{
  for(UInt i=0; i<NUM_REF_PIC_LIST_01; i++)
  {
    m_acYuvPred[i].clear();
  }
  m_cYuvPredTemp.clear();
  pcPredYuv->clear();

  if ( !bUseRes )
  {
    pcResiYuv->clear();
  }

  pcRecoYuv->clear();

  TComMv       cMvSrchRngLT;
  TComMv       cMvSrchRngRB;

  TComMv       cMvZero;
  TComMv       TempMv; //kolya

  TComMv       cMv[2];
  TComMv       cMvBi[2];
  TComMv       cMvTemp[2][33];

  Int          iNumPart    = pcCU->getNumPartitions();
  Int          iNumPredDir = pcCU->getSlice()->isInterP() ? 1 : 2;

  TComMv       cMvPred[2][33];

  TComMv       cMvPredBi[2][33];
  Int          aaiMvpIdxBi[2][33];

  Int          aaiMvpIdx[2][33];
  Int          aaiMvpNum[2][33];

  AMVPInfo     aacAMVPInfo[2][33];

  Int          iRefIdx[2]={0,0}; //If un-initialized, may cause SEGV in bi-directional prediction iterative stage.
  Int          iRefIdxBi[2] = {-1,-1};

  UInt         uiPartAddr;
  Int          iRoiWidth, iRoiHeight;

  UInt         uiMbBits[3] = {1, 1, 0};

  UInt         uiLastMode = 0;
  Int          iRefStart, iRefEnd;

  PartSize     ePartSize = pcCU->getPartitionSize( 0 );

  Int          bestBiPRefIdxL1 = 0;
  Int          bestBiPMvpL1 = 0;
  Distortion   biPDistTemp = std::numeric_limits<Distortion>::max();

  TComMvField cMvFieldNeighbours[MRG_MAX_NUM_CANDS << 1]; // double length for mv of both lists
  UChar uhInterDirNeighbours[MRG_MAX_NUM_CANDS];
  Int numValidMergeCand = 0 ;

  for ( Int iPartIdx = 0; iPartIdx < iNumPart; iPartIdx++ )
  {
    Distortion   uiCost[2] = { std::numeric_limits<Distortion>::max(), std::numeric_limits<Distortion>::max() };
    Distortion   uiCostBi  =   std::numeric_limits<Distortion>::max();
    Distortion   uiCostTemp;

    UInt         uiBits[3];
    UInt         uiBitsTemp;
    Distortion   bestBiPDist = std::numeric_limits<Distortion>::max();

    Distortion   uiCostTempL0[MAX_NUM_REF];
    for (Int iNumRef=0; iNumRef < MAX_NUM_REF; iNumRef++)
    {
      uiCostTempL0[iNumRef] = std::numeric_limits<Distortion>::max();
    }
    UInt         uiBitsTempL0[MAX_NUM_REF];

    TComMv       mvValidList1;
    Int          refIdxValidList1 = 0;
    UInt         bitsValidList1 = MAX_UINT;
    Distortion   costValidList1 = std::numeric_limits<Distortion>::max();

    xGetBlkBits( ePartSize, pcCU->getSlice()->isInterP(), iPartIdx, uiLastMode, uiMbBits);

    pcCU->getPartIndexAndSize( iPartIdx, uiPartAddr, iRoiWidth, iRoiHeight );

#if AMP_MRG
    Bool bTestNormalMC = true;

    if ( bUseMRG && pcCU->getWidth( 0 ) > 8 && iNumPart == 2 )
    {
      bTestNormalMC = false;
    }

    if (bTestNormalMC)
    {
#endif

    //  Uni-directional prediction
    for ( Int iRefList = 0; iRefList < iNumPredDir; iRefList++ )
    {
      RefPicList  eRefPicList = ( iRefList ? REF_PIC_LIST_1 : REF_PIC_LIST_0 );

      Int refPicNumber = pcCU->getSlice()->getNumRefIdx( eRefPicList );
      if ( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && eRefPicList == REF_PIC_LIST_0 )
      {
        refPicNumber--;
      }
      for ( Int iRefIdxTemp = 0; iRefIdxTemp < refPicNumber; iRefIdxTemp++ )
      {
        uiBitsTemp = uiMbBits[iRefList];
        if ( pcCU->getSlice()->getNumRefIdx(eRefPicList) > 1 )
        {
          uiBitsTemp += iRefIdxTemp+1;
          if ( iRefIdxTemp == pcCU->getSlice()->getNumRefIdx(eRefPicList)-1 )
          {
            uiBitsTemp--;
          }
        }
        xEstimateMvPredAMVP( pcCU, pcOrgYuv, iPartIdx, eRefPicList, iRefIdxTemp, cMvPred[iRefList][iRefIdxTemp], false, &biPDistTemp);
        aaiMvpIdx[iRefList][iRefIdxTemp] = pcCU->getMVPIdx(eRefPicList, uiPartAddr);
        aaiMvpNum[iRefList][iRefIdxTemp] = pcCU->getMVPNum(eRefPicList, uiPartAddr);

        if(pcCU->getSlice()->getMvdL1ZeroFlag() && iRefList==1 && biPDistTemp < bestBiPDist)
        {
          bestBiPDist = biPDistTemp;
          bestBiPMvpL1 = aaiMvpIdx[iRefList][iRefIdxTemp];
          bestBiPRefIdxL1 = iRefIdxTemp;
        }

        uiBitsTemp += m_auiMVPIdxCost[aaiMvpIdx[iRefList][iRefIdxTemp]][AMVP_MAX_NUM_CANDS];

        if ( m_pcEncCfg->getFastMEForGenBLowDelayEnabled() && iRefList == 1 )    // list 1
        {
          if ( pcCU->getSlice()->getList1IdxToList0Idx( iRefIdxTemp ) >= 0 )
          {
            cMvTemp[1][iRefIdxTemp] = cMvTemp[0][pcCU->getSlice()->getList1IdxToList0Idx( iRefIdxTemp )];
            uiCostTemp = uiCostTempL0[pcCU->getSlice()->getList1IdxToList0Idx( iRefIdxTemp )];
            /*first subtract the bit-rate part of the cost of the other list*/
            uiCostTemp -= m_pcRdCost->getCost( uiBitsTempL0[pcCU->getSlice()->getList1IdxToList0Idx( iRefIdxTemp )] );
            /*correct the bit-rate part of the current ref*/
            m_pcRdCost->setPredictor  ( cMvPred[iRefList][iRefIdxTemp] );
            uiBitsTemp += m_pcRdCost->getBitsOfVectorWithPredictor( cMvTemp[1][iRefIdxTemp].getHor(), cMvTemp[1][iRefIdxTemp].getVer() );
            /*calculate the correct cost*/
            uiCostTemp += m_pcRdCost->getCost( uiBitsTemp );
          }
          else
          {
            xMotionEstimation ( pcCU, pcOrgYuv, iPartIdx, eRefPicList, &cMvPred[iRefList][iRefIdxTemp], iRefIdxTemp, cMvTemp[iRefList][iRefIdxTemp], uiBitsTemp, uiCostTemp );
          }
        }
        else
        {
          xMotionEstimation ( pcCU, pcOrgYuv, iPartIdx, eRefPicList, &cMvPred[iRefList][iRefIdxTemp], iRefIdxTemp, cMvTemp[iRefList][iRefIdxTemp], uiBitsTemp, uiCostTemp );
        }
        xCopyAMVPInfo(pcCU->getCUMvField(eRefPicList)->getAMVPInfo(), &aacAMVPInfo[iRefList][iRefIdxTemp]); // must always be done ( also when AMVP_MODE = AM_NONE )
        xCheckBestMVP(pcCU, eRefPicList, cMvTemp[iRefList][iRefIdxTemp], cMvPred[iRefList][iRefIdxTemp], aaiMvpIdx[iRefList][iRefIdxTemp], uiBitsTemp, uiCostTemp);

        if ( iRefList <= 1 && iRefIdxTemp <= 1 && (ePartSize == SIZE_2NxN || ePartSize == SIZE_Nx2N) && pcCU->getWidth( 0 ) <= 16 )
        {
          iMVCandList[4*iRefList + 2*iRefIdxTemp + iPartIdx] = cMvTemp[iRefList][iRefIdxTemp];
        }
        if ( iRefList == 0 )
        {
          uiCostTempL0[iRefIdxTemp] = uiCostTemp;
          uiBitsTempL0[iRefIdxTemp] = uiBitsTemp;
        }
        if ( uiCostTemp < uiCost[iRefList] )
        {
          uiCost[iRefList] = uiCostTemp;
          uiBits[iRefList] = uiBitsTemp; // storing for bi-prediction

          // set motion
          cMv[iRefList]     = cMvTemp[iRefList][iRefIdxTemp];
          iRefIdx[iRefList] = iRefIdxTemp;
        }

        if ( iRefList == 1 && uiCostTemp < costValidList1 && pcCU->getSlice()->getList1IdxToList0Idx( iRefIdxTemp ) < 0 )
        {
          costValidList1 = uiCostTemp;
          bitsValidList1 = uiBitsTemp;

          // set motion
          mvValidList1     = cMvTemp[iRefList][iRefIdxTemp];
          refIdxValidList1 = iRefIdxTemp;
        }
      }
    }

    //  Bi-predictive Motion estimation
    if ( (pcCU->getSlice()->isInterB()) && (pcCU->isBipredRestriction(iPartIdx) == false) )
    {

      cMvBi[0] = cMv[0];            cMvBi[1] = cMv[1];
      iRefIdxBi[0] = iRefIdx[0];    iRefIdxBi[1] = iRefIdx[1];

      ::memcpy(cMvPredBi, cMvPred, sizeof(cMvPred));
      ::memcpy(aaiMvpIdxBi, aaiMvpIdx, sizeof(aaiMvpIdx));

      UInt uiMotBits[2];

      if(pcCU->getSlice()->getMvdL1ZeroFlag())
      {
        xCopyAMVPInfo(&aacAMVPInfo[1][bestBiPRefIdxL1], pcCU->getCUMvField(REF_PIC_LIST_1)->getAMVPInfo());
        pcCU->setMVPIdxSubParts( bestBiPMvpL1, REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
        aaiMvpIdxBi[1][bestBiPRefIdxL1] = bestBiPMvpL1;
        cMvPredBi[1][bestBiPRefIdxL1]   = pcCU->getCUMvField(REF_PIC_LIST_1)->getAMVPInfo()->m_acMvCand[bestBiPMvpL1];

        cMvBi[1] = cMvPredBi[1][bestBiPRefIdxL1];
        iRefIdxBi[1] = bestBiPRefIdxL1;
        if( pcCU->getSlice()->getUseIntegerMv() )
        {
          cMvBi[1] = (cMvBi[1]>>2)<<2;
        }
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( cMvBi[1], ePartSize, uiPartAddr, 0, iPartIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( iRefIdxBi[1], ePartSize, uiPartAddr, 0, iPartIdx );
        TComYuv* pcYuvPred = &m_acYuvPred[REF_PIC_LIST_1];
        motionCompensation( pcCU, pcYuvPred, REF_PIC_LIST_1, iPartIdx );

        uiMotBits[0] = uiBits[0] - uiMbBits[0];
        uiMotBits[1] = uiMbBits[1];

        if ( pcCU->getSlice()->getNumRefIdx(REF_PIC_LIST_1) > 1 )
        {
          uiMotBits[1] += bestBiPRefIdxL1+1;
          if ( bestBiPRefIdxL1 == pcCU->getSlice()->getNumRefIdx(REF_PIC_LIST_1)-1 )
          {
            uiMotBits[1]--;
          }
        }

        uiMotBits[1] += m_auiMVPIdxCost[aaiMvpIdxBi[1][bestBiPRefIdxL1]][AMVP_MAX_NUM_CANDS];

        uiBits[2] = uiMbBits[2] + uiMotBits[0] + uiMotBits[1];

        cMvTemp[1][bestBiPRefIdxL1] = cMvBi[1];
      }
      else
      {
        uiMotBits[0] = uiBits[0] - uiMbBits[0];
        uiMotBits[1] = uiBits[1] - uiMbBits[1];
        uiBits[2] = uiMbBits[2] + uiMotBits[0] + uiMotBits[1];
      }

      // 4-times iteration (default)
      Int iNumIter = 4;

      // fast encoder setting: only one iteration
      if ( m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE1 || m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE2 || pcCU->getSlice()->getMvdL1ZeroFlag() )
      {
        iNumIter = 1;
      }

      for ( Int iIter = 0; iIter < iNumIter; iIter++ )
      {
        Int         iRefList    = iIter % 2;

        if ( m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE1 || m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE2 )
        {
          if( uiCost[0] <= uiCost[1] )
          {
            iRefList = 1;
          }
          else
          {
            iRefList = 0;
          }
        }
        else if ( iIter == 0 )
        {
          iRefList = 0;
        }
        if ( iIter == 0 && !pcCU->getSlice()->getMvdL1ZeroFlag())
        {
          pcCU->getCUMvField(RefPicList(1-iRefList))->setAllMv( cMv[1-iRefList], ePartSize, uiPartAddr, 0, iPartIdx );
          pcCU->getCUMvField(RefPicList(1-iRefList))->setAllRefIdx( iRefIdx[1-iRefList], ePartSize, uiPartAddr, 0, iPartIdx );
          TComYuv*  pcYuvPred = &m_acYuvPred[1-iRefList];
          motionCompensation ( pcCU, pcYuvPred, RefPicList(1-iRefList), iPartIdx );
        }

        RefPicList  eRefPicList = ( iRefList ? REF_PIC_LIST_1 : REF_PIC_LIST_0 );

        if(pcCU->getSlice()->getMvdL1ZeroFlag())
        {
          iRefList = 0;
          eRefPicList = REF_PIC_LIST_0;
        }

        Bool bChanged = false;

        iRefStart = 0;
        iRefEnd   = pcCU->getSlice()->getNumRefIdx(eRefPicList)-1;
        if ( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && eRefPicList == REF_PIC_LIST_0 )
        {
          iRefEnd--;
        }

        for ( Int iRefIdxTemp = iRefStart; iRefIdxTemp <= iRefEnd; iRefIdxTemp++ )
        {
          uiBitsTemp = uiMbBits[2] + uiMotBits[1-iRefList];
          if ( pcCU->getSlice()->getNumRefIdx(eRefPicList) > 1 )
          {
            uiBitsTemp += iRefIdxTemp+1;
            if ( iRefIdxTemp == pcCU->getSlice()->getNumRefIdx(eRefPicList)-1 )
            {
              uiBitsTemp--;
            }
          }
          uiBitsTemp += m_auiMVPIdxCost[aaiMvpIdxBi[iRefList][iRefIdxTemp]][AMVP_MAX_NUM_CANDS];
          // call ME
          xMotionEstimation ( pcCU, pcOrgYuv, iPartIdx, eRefPicList, &cMvPredBi[iRefList][iRefIdxTemp], iRefIdxTemp, cMvTemp[iRefList][iRefIdxTemp], uiBitsTemp, uiCostTemp, true );

          xCopyAMVPInfo(&aacAMVPInfo[iRefList][iRefIdxTemp], pcCU->getCUMvField(eRefPicList)->getAMVPInfo());
          xCheckBestMVP(pcCU, eRefPicList, cMvTemp[iRefList][iRefIdxTemp], cMvPredBi[iRefList][iRefIdxTemp], aaiMvpIdxBi[iRefList][iRefIdxTemp], uiBitsTemp, uiCostTemp);

          if ( uiCostTemp < uiCostBi )
          {
            bChanged = true;

            cMvBi[iRefList]     = cMvTemp[iRefList][iRefIdxTemp];
            iRefIdxBi[iRefList] = iRefIdxTemp;

            uiCostBi            = uiCostTemp;
            uiMotBits[iRefList] = uiBitsTemp - uiMbBits[2] - uiMotBits[1-iRefList];
            uiBits[2]           = uiBitsTemp;

            if(iNumIter!=1)
            {
              //  Set motion
              pcCU->getCUMvField( eRefPicList )->setAllMv( cMvBi[iRefList], ePartSize, uiPartAddr, 0, iPartIdx );
              pcCU->getCUMvField( eRefPicList )->setAllRefIdx( iRefIdxBi[iRefList], ePartSize, uiPartAddr, 0, iPartIdx );

              TComYuv* pcYuvPred = &m_acYuvPred[iRefList];
              motionCompensation( pcCU, pcYuvPred, eRefPicList, iPartIdx );
            }
          }
        } // for loop-iRefIdxTemp

        if ( !bChanged )
        {
          Bool b8x8BiPredRestricted = pcCU->is8x8BipredRestriction(cMvBi[0],cMvBi[1],iRefIdxBi[0],iRefIdxBi[1]);
          if ( uiCostBi <= uiCost[0] && uiCostBi <= uiCost[1] && !pcCU->isBipredRestriction(iPartIdx,b8x8BiPredRestricted) )
          {
            xCopyAMVPInfo(&aacAMVPInfo[0][iRefIdxBi[0]], pcCU->getCUMvField(REF_PIC_LIST_0)->getAMVPInfo());
            xCheckBestMVP(pcCU, REF_PIC_LIST_0, cMvBi[0], cMvPredBi[0][iRefIdxBi[0]], aaiMvpIdxBi[0][iRefIdxBi[0]], uiBits[2], uiCostBi);
            if(!pcCU->getSlice()->getMvdL1ZeroFlag())
            {
              xCopyAMVPInfo(&aacAMVPInfo[1][iRefIdxBi[1]], pcCU->getCUMvField(REF_PIC_LIST_1)->getAMVPInfo());
              xCheckBestMVP(pcCU, REF_PIC_LIST_1, cMvBi[1], cMvPredBi[1][iRefIdxBi[1]], aaiMvpIdxBi[1][iRefIdxBi[1]], uiBits[2], uiCostBi);
            }
          }
          break;
        }
      } // for loop-iter
    } // if (B_SLICE)

#if AMP_MRG
    } //end if bTestNormalMC
#endif
    //  Clear Motion Field
    pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvField( TComMvField(), ePartSize, uiPartAddr, 0, iPartIdx );
    pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvField( TComMvField(), ePartSize, uiPartAddr, 0, iPartIdx );
    pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvd    ( cMvZero,       ePartSize, uiPartAddr, 0, iPartIdx );
    pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvd    ( cMvZero,       ePartSize, uiPartAddr, 0, iPartIdx );

    pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
    pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
    pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
    pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));

    UInt uiMEBits = 0;
    // Set Motion Field_
    cMv[1] = mvValidList1;
    iRefIdx[1] = refIdxValidList1;
    uiBits[1] = bitsValidList1;
    uiCost[1] = costValidList1;

#if AMP_MRG
    if (bTestNormalMC)
    {
#endif
     Bool b8x8BiPredRestricted = pcCU->is8x8BipredRestriction(cMvBi[0],cMvBi[1],iRefIdxBi[0],iRefIdxBi[1]);
     if ( uiCostBi <= uiCost[0] && uiCostBi <= uiCost[1] && !pcCU->isBipredRestriction(iPartIdx,b8x8BiPredRestricted) )
    {
      uiLastMode = 2;
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMv( cMvBi[0], ePartSize, uiPartAddr, 0, iPartIdx );
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllRefIdx( iRefIdxBi[0], ePartSize, uiPartAddr, 0, iPartIdx );
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMv( cMvBi[1], ePartSize, uiPartAddr, 0, iPartIdx );
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllRefIdx( iRefIdxBi[1], ePartSize, uiPartAddr, 0, iPartIdx );

      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        TempMv = (cMvBi[0]>>2) - (cMvPredBi[0][iRefIdxBi[0]]>>2);
      }
      else
      {
        TempMv = cMvBi[0] - cMvPredBi[0][iRefIdxBi[0]];
      }
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvd    ( TempMv,                 ePartSize, uiPartAddr, 0, iPartIdx );

      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        TempMv = (cMvBi[1]>>2) - (cMvPredBi[1][iRefIdxBi[1]]>>2);
      }
      else
      {
        TempMv = cMvBi[1] - cMvPredBi[1][iRefIdxBi[1]];
      }
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvd    ( TempMv,                 ePartSize, uiPartAddr, 0, iPartIdx );

      pcCU->setInterDirSubParts( 3, uiPartAddr, iPartIdx, pcCU->getDepth(0) );

      pcCU->setMVPIdxSubParts( aaiMvpIdxBi[0][iRefIdxBi[0]], REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      pcCU->setMVPNumSubParts( aaiMvpNum[0][iRefIdxBi[0]], REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      pcCU->setMVPIdxSubParts( aaiMvpIdxBi[1][iRefIdxBi[1]], REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      pcCU->setMVPNumSubParts( aaiMvpNum[1][iRefIdxBi[1]], REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));

      uiMEBits = uiBits[2];
    }
    else if ( uiCost[0] <= uiCost[1] )
    {
      uiLastMode = 0;
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMv( cMv[0], ePartSize, uiPartAddr, 0, iPartIdx );
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllRefIdx( iRefIdx[0], ePartSize, uiPartAddr, 0, iPartIdx );

      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        TempMv = (cMv[0]>>2) - (cMvPred[0][iRefIdx[0]]>>2);
      }
      else
      {
        TempMv = cMv[0] - cMvPred[0][iRefIdx[0]];
      }
      pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvd    ( TempMv,                 ePartSize, uiPartAddr, 0, iPartIdx );

      pcCU->setInterDirSubParts( 1, uiPartAddr, iPartIdx, pcCU->getDepth(0) );

      pcCU->setMVPIdxSubParts( aaiMvpIdx[0][iRefIdx[0]], REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      pcCU->setMVPNumSubParts( aaiMvpNum[0][iRefIdx[0]], REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));

      uiMEBits = uiBits[0];
    }
    else
    {
      uiLastMode = 1;
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMv( cMv[1], ePartSize, uiPartAddr, 0, iPartIdx );
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllRefIdx( iRefIdx[1], ePartSize, uiPartAddr, 0, iPartIdx );

      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        TempMv = (cMv[1]>>2) - (cMvPred[1][iRefIdx[1]]>>2);
      }
      else
      {
        TempMv = cMv[1] - cMvPred[1][iRefIdx[1]];
      }
      pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvd    ( TempMv,                 ePartSize, uiPartAddr, 0, iPartIdx );

      pcCU->setInterDirSubParts( 2, uiPartAddr, iPartIdx, pcCU->getDepth(0) );

      pcCU->setMVPIdxSubParts( aaiMvpIdx[1][iRefIdx[1]], REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      pcCU->setMVPNumSubParts( aaiMvpNum[1][iRefIdx[1]], REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));

      uiMEBits = uiBits[1];
    }
#if AMP_MRG
    } // end if bTestNormalMC
#endif

    if ( pcCU->getPartitionSize( uiPartAddr ) != SIZE_2Nx2N )
    {
      UInt uiMRGInterDir = 0;
      TComMvField cMRGMvField[2];
      UInt uiMRGIndex = 0;

      UInt uiMEInterDir = 0;
      TComMvField cMEMvField[2];

      m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(uiPartAddr) );

#if AMP_MRG
      // calculate ME cost
      Distortion uiMEError = std::numeric_limits<Distortion>::max();
      Distortion uiMECost  = std::numeric_limits<Distortion>::max();

      if (bTestNormalMC)
      {
        xGetInterPredictionError( pcCU, pcOrgYuv, iPartIdx, uiMEError, m_pcEncCfg->getUseHADME() );
        uiMECost = uiMEError + m_pcRdCost->getCost( uiMEBits );
      }
#else
      // calculate ME cost
      Distortion uiMEError = std::numeric_limits<Distortion>::max();
      xGetInterPredictionError( pcCU, pcOrgYuv, iPartIdx, uiMEError, m_pcEncCfg->getUseHADME() );
      Distortion uiMECost = uiMEError + m_pcRdCost->getCost( uiMEBits );
#endif
      // save ME result.
      uiMEInterDir = pcCU->getInterDir( uiPartAddr );
      TComDataCU::getMvField( pcCU, uiPartAddr, REF_PIC_LIST_0, cMEMvField[0] );
      TComDataCU::getMvField( pcCU, uiPartAddr, REF_PIC_LIST_1, cMEMvField[1] );

      // find Merge result
      Distortion uiMRGCost = std::numeric_limits<Distortion>::max();
      pcCU->setMergeFlagSubParts( true, uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
      xMergeEstimation( pcCU, pcOrgYuv, iPartIdx, uiMRGInterDir, cMRGMvField, uiMRGIndex, uiMRGCost, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand);

      if ( uiMRGCost < uiMECost )
      {
        // set Merge result
        pcCU->setMergeFlagSubParts ( true,          uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
        pcCU->setMergeIndexSubParts( uiMRGIndex,    uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
        pcCU->setInterDirSubParts  ( uiMRGInterDir, uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvField[0], ePartSize, uiPartAddr, 0, iPartIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvField[1], ePartSize, uiPartAddr, 0, iPartIdx );

        pcCU->getCUMvField(REF_PIC_LIST_0)->setAllMvd    ( cMvZero,            ePartSize, uiPartAddr, 0, iPartIdx );
        pcCU->getCUMvField(REF_PIC_LIST_1)->setAllMvd    ( cMvZero,            ePartSize, uiPartAddr, 0, iPartIdx );

        pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
        pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_0, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
        pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
        pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_1, uiPartAddr, iPartIdx, pcCU->getDepth(uiPartAddr));
      }
      else if( uiMECost < std::numeric_limits<Distortion>::max() )
      {
        // set ME result
        pcCU->setMergeFlagSubParts( false,        uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
        pcCU->setInterDirSubParts ( uiMEInterDir, uiPartAddr, iPartIdx, pcCU->getDepth( uiPartAddr ) );
        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMEMvField[0], ePartSize, uiPartAddr, 0, iPartIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMEMvField[1], ePartSize, uiPartAddr, 0, iPartIdx );
      }
      else
      {
        return false;
      }
    }

#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, iPartIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif

    //  MC
    motionCompensation ( pcCU, pcPredYuv, REF_PIC_LIST_X, iPartIdx );

  } //  end of for ( Int iPartIdx = 0; iPartIdx < iNumPart; iPartIdx++ )

  setWpScalingDistParam( pcCU, -1, REF_PIC_LIST_X );

  return true;
}


// AMVP
Void TEncSearch::xEstimateMvPredAMVP( TComDataCU* pcCU, TComYuv* pcOrgYuv, UInt uiPartIdx, RefPicList eRefPicList, Int iRefIdx, TComMv& rcMvPred, Bool bFilled, Distortion* puiDistBiP )
{
  AMVPInfo*  pcAMVPInfo = pcCU->getCUMvField(eRefPicList)->getAMVPInfo();

  TComMv     cBestMv;
  Int        iBestIdx   = 0;
  TComMv     cZeroMv;
  TComMv     cMvPred;
  Distortion uiBestCost = std::numeric_limits<Distortion>::max();
  UInt       uiPartAddr = 0;
  Int        iRoiWidth, iRoiHeight;
  Int        i;
  Int        minMVPCand;
  Int        maxMVPCand;

  pcCU->getPartIndexAndSize( uiPartIdx, uiPartAddr, iRoiWidth, iRoiHeight );
  // Fill the MV Candidates
  if (!bFilled)
  {
    pcCU->fillMvpCand( uiPartIdx, uiPartAddr, eRefPicList, iRefIdx, pcAMVPInfo );
  }
  // initialize Mvp index & Mvp
#if MCTS_ENC_CHECK
  if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile() && (pcAMVPInfo->numSpatialMVPCandidates < pcAMVPInfo->iN))
  {
    iBestIdx    = (pcAMVPInfo->numSpatialMVPCandidates == 0) ? 1 : 0;
    cBestMv     = pcAMVPInfo->m_acMvCand[(pcAMVPInfo->numSpatialMVPCandidates == 0) ? 1 : 0];
    minMVPCand  = (pcAMVPInfo->numSpatialMVPCandidates == 0) ? 1 : 0;
    maxMVPCand  = (pcAMVPInfo->numSpatialMVPCandidates == 0) ? pcAMVPInfo->iN : 1;
  }
  else
  {
    iBestIdx = 0;
    cBestMv  = pcAMVPInfo->m_acMvCand[0];
    minMVPCand  = 0;
    maxMVPCand  = pcAMVPInfo->iN;
  }
#else
  iBestIdx = 0;
  cBestMv  = pcAMVPInfo->m_acMvCand[0];
  minMVPCand  = 0;
  maxMVPCand  = pcAMVPInfo->iN;
#endif
  if (pcAMVPInfo->iN <= 1)
  {
    rcMvPred = cBestMv;

    pcCU->setMVPIdxSubParts( iBestIdx, eRefPicList, uiPartAddr, uiPartIdx, pcCU->getDepth(uiPartAddr));
    pcCU->setMVPNumSubParts( pcAMVPInfo->iN, eRefPicList, uiPartAddr, uiPartIdx, pcCU->getDepth(uiPartAddr));

    if(pcCU->getSlice()->getMvdL1ZeroFlag() && eRefPicList==REF_PIC_LIST_1)
    {
      (*puiDistBiP) = xGetTemplateCost( pcCU, uiPartAddr, pcOrgYuv, &m_cYuvPredTemp, rcMvPred, 0, AMVP_MAX_NUM_CANDS, eRefPicList, iRefIdx, iRoiWidth, iRoiHeight);
    }
    return;
  }

  if (bFilled)
  {
    assert(pcCU->getMVPIdx(eRefPicList,uiPartAddr) >= 0);
    rcMvPred = pcAMVPInfo->m_acMvCand[pcCU->getMVPIdx(eRefPicList,uiPartAddr)];
    return;
  }

  m_cYuvPredTemp.clear();
  //-- Check Minimum Cost.
  for ( i = minMVPCand ; i < maxMVPCand; i++)
  {
    Distortion uiTmpCost;
    uiTmpCost = xGetTemplateCost( pcCU, uiPartAddr, pcOrgYuv, &m_cYuvPredTemp, pcAMVPInfo->m_acMvCand[i], i, AMVP_MAX_NUM_CANDS, eRefPicList, iRefIdx, iRoiWidth, iRoiHeight);
    if ( uiBestCost > uiTmpCost )
    {
      uiBestCost = uiTmpCost;
      cBestMv   = pcAMVPInfo->m_acMvCand[i];
      iBestIdx  = i;
      (*puiDistBiP) = uiTmpCost;
    }
  }

  m_cYuvPredTemp.clear();

  // Setting Best MVP
  rcMvPred = cBestMv;
  pcCU->setMVPIdxSubParts( iBestIdx, eRefPicList, uiPartAddr, uiPartIdx, pcCU->getDepth(uiPartAddr));
  pcCU->setMVPNumSubParts( pcAMVPInfo->iN, eRefPicList, uiPartAddr, uiPartIdx, pcCU->getDepth(uiPartAddr));
  return;
}

UInt TEncSearch::xGetMvpIdxBits(Int iIdx, Int iNum)
{
  assert(iIdx >= 0 && iNum >= 0 && iIdx < iNum);

  if (iNum == 1)
  {
    return 0;
  }

  UInt uiLength = 1;
  Int iTemp = iIdx;
  if ( iTemp == 0 )
  {
    return uiLength;
  }

  Bool bCodeLast = ( iNum-1 > iTemp );

  uiLength += (iTemp-1);

  if( bCodeLast )
  {
    uiLength++;
  }

  return uiLength;
}

Void TEncSearch::xGetBlkBits( PartSize eCUMode, Bool bPSlice, Int iPartIdx, UInt uiLastMode, UInt uiBlkBit[3])
{
  if ( eCUMode == SIZE_2Nx2N )
  {
    uiBlkBit[0] = (! bPSlice) ? 3 : 1;
    uiBlkBit[1] = 3;
    uiBlkBit[2] = 5;
  }
  else if ( (eCUMode == SIZE_2NxN || eCUMode == SIZE_2NxnU) || eCUMode == SIZE_2NxnD )
  {
    UInt aauiMbBits[2][3][3] = { { {0,0,3}, {0,0,0}, {0,0,0} } , { {5,7,7}, {7,5,7}, {9-3,9-3,9-3} } };
    if ( bPSlice )
    {
      uiBlkBit[0] = 3;
      uiBlkBit[1] = 0;
      uiBlkBit[2] = 0;
    }
    else
    {
      ::memcpy( uiBlkBit, aauiMbBits[iPartIdx][uiLastMode], 3*sizeof(UInt) );
    }
  }
  else if ( (eCUMode == SIZE_Nx2N || eCUMode == SIZE_nLx2N) || eCUMode == SIZE_nRx2N )
  {
    UInt aauiMbBits[2][3][3] = { { {0,2,3}, {0,0,0}, {0,0,0} } , { {5,7,7}, {7-2,7-2,9-2}, {9-3,9-3,9-3} } };
    if ( bPSlice )
    {
      uiBlkBit[0] = 3;
      uiBlkBit[1] = 0;
      uiBlkBit[2] = 0;
    }
    else
    {
      ::memcpy( uiBlkBit, aauiMbBits[iPartIdx][uiLastMode], 3*sizeof(UInt) );
    }
  }
  else if ( eCUMode == SIZE_NxN )
  {
    uiBlkBit[0] = (! bPSlice) ? 3 : 1;
    uiBlkBit[1] = 3;
    uiBlkBit[2] = 5;
  }
  else
  {
    printf("Wrong!\n");
    assert( 0 );
  }
}

Void TEncSearch::xCopyAMVPInfo (AMVPInfo* pSrc, AMVPInfo* pDst)
{
  pDst->iN = pSrc->iN;
  for (Int i = 0; i < pSrc->iN; i++)
  {
    pDst->m_acMvCand[i] = pSrc->m_acMvCand[i];
  }
}

Void TEncSearch::xCheckBestMVP ( TComDataCU* pcCU, RefPicList eRefPicList, TComMv cMv, TComMv& rcMvPred, Int& riMVPIdx, UInt& ruiBits, Distortion& ruiCost )
{
  AMVPInfo* pcAMVPInfo = pcCU->getCUMvField(eRefPicList)->getAMVPInfo();
  if ( !pcCU->getSlice()->getUseIntegerMv() )
  {
    assert( pcAMVPInfo->m_acMvCand[riMVPIdx] == rcMvPred );
  }

  if (pcAMVPInfo->iN < 2)
  {
    return;
  }

  m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(0) );
  m_pcRdCost->setCostScale ( 0    );

  Int iBestMVPIdx = riMVPIdx;

  m_pcRdCost->setPredictor( rcMvPred );
  Int iOrgMvBits  = m_pcRdCost->getBitsOfVectorWithPredictor(cMv.getHor(), cMv.getVer());
  iOrgMvBits += m_auiMVPIdxCost[riMVPIdx][AMVP_MAX_NUM_CANDS];
  Int iBestMvBits = iOrgMvBits;

#if MCTS_ENC_CHECK
  Int minMVPCand = 0;
  Int maxMVPCand = pcAMVPInfo->iN;

  if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
  {
    minMVPCand = (pcAMVPInfo->numSpatialMVPCandidates == 0) ? 1 : 0;
    maxMVPCand = (pcAMVPInfo->numSpatialMVPCandidates == 0) ? pcAMVPInfo->iN : 1;
  }
  for (Int iMVPIdx = minMVPCand; iMVPIdx < maxMVPCand; iMVPIdx++)
#else
  for (Int iMVPIdx = 0; iMVPIdx < pcAMVPInfo->iN; iMVPIdx++)
#endif
  {
    if (iMVPIdx == riMVPIdx)
    {
      continue;
    }

    m_pcRdCost->setPredictor( pcAMVPInfo->m_acMvCand[iMVPIdx] );

    Int iMvBits = m_pcRdCost->getBitsOfVectorWithPredictor(cMv.getHor(), cMv.getVer());
    iMvBits += m_auiMVPIdxCost[iMVPIdx][AMVP_MAX_NUM_CANDS];

    if (iMvBits < iBestMvBits)
    {
      iBestMvBits = iMvBits;
      iBestMVPIdx = iMVPIdx;
    }
  }

  if (iBestMVPIdx != riMVPIdx)  //if changed
  {
    rcMvPred = pcAMVPInfo->m_acMvCand[iBestMVPIdx];

    riMVPIdx = iBestMVPIdx;
    UInt uiOrgBits = ruiBits;
    ruiBits = uiOrgBits - iOrgMvBits + iBestMvBits;
    ruiCost = (ruiCost - m_pcRdCost->getCost( uiOrgBits ))  + m_pcRdCost->getCost( ruiBits );
  }
}


Distortion TEncSearch::xGetTemplateCost( TComDataCU* pcCU,
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
                                         )
{
  Distortion uiCost = std::numeric_limits<Distortion>::max();

  TComPicYuv* pcPicYuvRef = pcCU->getSlice()->getRefPic( eRefPicList, iRefIdx )->getPicYuvRec();

  pcCU->clipMv( cMvCand );

  // prediction pattern
  if ( pcCU->getSlice()->testWeightPred() && pcCU->getSlice()->getSliceType()==P_SLICE )
  {
    xPredInterBlk( COMPONENT_Y, pcCU, pcPicYuvRef, uiPartAddr, &cMvCand, iSizeX, iSizeY, pcTemplateCand, true, pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
  }
  else
  {
    xPredInterBlk( COMPONENT_Y, pcCU, pcPicYuvRef, uiPartAddr, &cMvCand, iSizeX, iSizeY, pcTemplateCand, false, pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
  }

  if ( pcCU->getSlice()->testWeightPred() && pcCU->getSlice()->getSliceType()==P_SLICE )
  {
    xWeightedPredictionUni( pcCU, pcTemplateCand, uiPartAddr, iSizeX, iSizeY, eRefPicList, pcTemplateCand, iRefIdx );
  }

  // calc distortion

  uiCost = m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA), pcTemplateCand->getAddr(COMPONENT_Y, uiPartAddr), pcTemplateCand->getStride(COMPONENT_Y), pcOrgYuv->getAddr(COMPONENT_Y, uiPartAddr), pcOrgYuv->getStride(COMPONENT_Y), iSizeX, iSizeY, COMPONENT_Y, DF_SAD );
  uiCost = (UInt) m_pcRdCost->calcRdCost( m_auiMVPIdxCost[iMVPIdx][iMVPNum], uiCost, DF_SAD );
  return uiCost;
}


Void TEncSearch::xMotionEstimation( TComDataCU* pcCU, TComYuv* pcYuvOrg, Int iPartIdx, RefPicList eRefPicList, TComMv* pcMvPred, Int iRefIdxPred, TComMv& rcMv, UInt& ruiBits, Distortion& ruiCost, Bool bBi  )
{
  UInt          uiPartAddr;
  Int           iRoiWidth;
  Int           iRoiHeight;

  TComMv        cMvHalf, cMvQter;
  TComMv        cMvSrchRngLT;
  TComMv        cMvSrchRngRB;

  TComYuv*      pcYuv = pcYuvOrg;

  assert(eRefPicList < MAX_NUM_REF_LIST_ADAPT_SR && iRefIdxPred<Int(MAX_IDX_ADAPT_SR));
  m_iSearchRange = m_aaiAdaptSR[eRefPicList][iRefIdxPred];

  Int           iSrchRng      = ( bBi ? m_bipredSearchRange : m_iSearchRange );
  TComPattern   cPattern;

  Double        fWeight       = 1.0;

  pcCU->getPartIndexAndSize( iPartIdx, uiPartAddr, iRoiWidth, iRoiHeight );

  if ( bBi ) // Bipredictive ME
  {
    TComYuv*  pcYuvOther = &m_acYuvPred[1-(Int)eRefPicList];
    pcYuv                = &m_cYuvPredTemp;

    pcYuvOrg->copyPartToPartYuv( pcYuv, uiPartAddr, iRoiWidth, iRoiHeight );

    pcYuv->removeHighFreq( pcYuvOther, uiPartAddr, iRoiWidth, iRoiHeight, pcCU->getSlice()->getSPS()->getBitDepths().recon, m_pcEncCfg->getClipForBiPredMeEnabled() );

    fWeight = 0.5;
  }
  m_cDistParam.bIsBiPred = bBi;

  //  Search key pattern initialization
#if MCTS_ENC_CHECK
  Int roiPosX, roiPosY; 
  Int roiW, roiH;
  pcCU->getPartPosition(iPartIdx, roiPosX, roiPosY, roiW, roiH);
  assert(roiW == iRoiWidth);
  assert(roiH == iRoiHeight);
  cPattern.initPattern( pcYuv->getAddr(COMPONENT_Y, uiPartAddr),
                        iRoiWidth,
                        iRoiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA),
                        roiPosX,
                        roiPosY);
  xInitTileBorders(pcCU, &cPattern);
#else
  cPattern.initPattern( pcYuv->getAddr  ( COMPONENT_Y, uiPartAddr ),
                        iRoiWidth,
                        iRoiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
#endif

  Pel*        piRefY      = pcCU->getSlice()->getRefPic( eRefPicList, iRefIdxPred )->getPicYuvRec()->getAddr( COMPONENT_Y, pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + uiPartAddr );
  Int         iRefStride  = pcCU->getSlice()->getRefPic( eRefPicList, iRefIdxPred )->getPicYuvRec()->getStride(COMPONENT_Y);

  TComMv      cMvPred = *pcMvPred;

  if ( bBi )
  {
#if MCTS_ENC_CHECK
    xSetSearchRange(pcCU, rcMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB, &cPattern);
#else
    xSetSearchRange(pcCU, rcMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB);
#endif
  }
  else
  {
#if MCTS_ENC_CHECK
    xSetSearchRange(pcCU, cMvPred, iSrchRng, cMvSrchRngLT, cMvSrchRngRB, &cPattern);
#else
    xSetSearchRange(pcCU, cMvPred, iSrchRng, cMvSrchRngLT, cMvSrchRngRB);
#endif
  }

  m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(uiPartAddr) );

  m_pcRdCost->setPredictor  ( *pcMvPred );
  m_pcRdCost->setCostScale  ( 2 );

  setWpScalingDistParam( pcCU, iRefIdxPred, eRefPicList );

  m_currRefPicList = eRefPicList;
  m_currRefPicIndex = iRefIdxPred;
  m_bSkipFracME = false;

  if ( pcCU->getSlice()->getUseIntegerMv() )
  {
    m_bSkipFracME = true;
  }

  //  Do integer search
  if ( (m_motionEstimationSearchMethod==MESEARCH_FULL) || bBi )
  {
    xPatternSearch      ( &cPattern, piRefY, iRefStride, &cMvSrchRngLT, &cMvSrchRngRB, rcMv, ruiCost );
  }
  else
  {
    rcMv = *pcMvPred;
    const TComMv *pIntegerMv2Nx2NPred=0;
    if (pcCU->getPartitionSize(0) != SIZE_2Nx2N || pcCU->getDepth(0) != 0)
    {
      pIntegerMv2Nx2NPred = &(m_integerMv2Nx2N[eRefPicList][iRefIdxPred]);
    }
    xPatternSearchFast  ( pcCU, &cPattern, piRefY, iRefStride, &cMvSrchRngLT, &cMvSrchRngRB, rcMv, ruiCost, pIntegerMv2Nx2NPred );
    if (pcCU->getPartitionSize(0) == SIZE_2Nx2N)
    {
      m_integerMv2Nx2N[eRefPicList][iRefIdxPred] = rcMv;
    }
  }

  m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(uiPartAddr) );
  m_pcRdCost->setCostScale ( 1 );

  const Bool bIsLosslessCoded = pcCU->getCUTransquantBypass(uiPartAddr) != 0;
  xPatternSearchFracDIF( bIsLosslessCoded, pcCU, &cPattern, piRefY, iRefStride, &rcMv, cMvHalf, cMvQter, ruiCost );

  m_pcRdCost->setCostScale( 0 );
  rcMv <<= 2;
  rcMv += (cMvHalf <<= 1);
  rcMv +=  cMvQter;

  UInt uiMvBits = m_pcRdCost->getBitsOfVectorWithPredictor( rcMv.getHor(), rcMv.getVer() );

  ruiBits      += uiMvBits;
  ruiCost       = (Distortion)( floor( fWeight * ( (Double)ruiCost - (Double)m_pcRdCost->getCost( uiMvBits ) ) ) + (Double)m_pcRdCost->getCost( ruiBits ) );
}

#if MCTS_ENC_CHECK
Void TEncSearch::xInitTileBorders(const TComDataCU* const pcCU, TComPattern* pcPatternKey)
{
  if (m_pcEncCfg->getTMCTSSEITileConstraint())
  {
    UInt  tileXPosInCtus = 0;
    UInt  tileYPosInCtus = 0;
    UInt  tileWidthtInCtus = 0;
    UInt  tileHeightInCtus = 0;

    getTilePosition(pcCU, tileXPosInCtus, tileYPosInCtus, tileWidthtInCtus, tileHeightInCtus);

    const Int  ctuLength = pcCU->getPic()->getPicSym()->getSPS().getMaxCUWidth();

    // tile position in full pels
    const Int tileLeftTopPelPosX = ctuLength * tileXPosInCtus;
    const Int tileLeftTopPelPosY = ctuLength * tileYPosInCtus;
    const Int tileRightBottomPelPosX = ((tileWidthtInCtus + tileXPosInCtus) * ctuLength) - 1;
    const Int tileRightBottomPelPosY = ((tileHeightInCtus + tileYPosInCtus) * ctuLength) - 1;

    pcPatternKey->setTileBorders (tileLeftTopPelPosX,tileLeftTopPelPosY,tileRightBottomPelPosX,tileRightBottomPelPosY);
  }
}
#endif


Void TEncSearch::xSetSearchRange ( const TComDataCU* const pcCU, const TComMv& cMvPred, const Int iSrchRng,
#if MCTS_ENC_CHECK
                                   TComMv& rcMvSrchRngLT, TComMv& rcMvSrchRngRB, const TComPattern* const pcPatternKey )
#else
                                   TComMv& rcMvSrchRngLT, TComMv& rcMvSrchRngRB )
#endif
{
  Int  iMvShift = 2;
  TComMv cTmpMvPred = cMvPred;
  pcCU->clipMv( cTmpMvPred );

#if MCTS_ENC_CHECK
  if (m_pcEncCfg->getTMCTSSEITileConstraint())
  {
    const Int lRangeXLeft = max(cTmpMvPred.getHor() - (iSrchRng << iMvShift), (pcPatternKey->getTileLeftTopPelPosX() - pcPatternKey->getROIYPosX()) << iMvShift);
    const Int lRangeYTop = max(cTmpMvPred.getVer() - (iSrchRng << iMvShift), (pcPatternKey->getTileLeftTopPelPosY() - pcPatternKey->getROIYPosY()) << iMvShift);
    const Int lRangeXRight = min(cTmpMvPred.getHor() + (iSrchRng << iMvShift), (pcPatternKey->getTileRightBottomPelPosX() - (pcPatternKey->getROIYPosX() + pcPatternKey->getROIYWidth())) << iMvShift);
    const Int lRangeYBottom = min(cTmpMvPred.getVer() + (iSrchRng << iMvShift), (pcPatternKey->getTileRightBottomPelPosY() - (pcPatternKey->getROIYPosY() + pcPatternKey->getROIYHeight())) << iMvShift);

    rcMvSrchRngLT.setHor(lRangeXLeft);
    rcMvSrchRngLT.setVer(lRangeYTop);

    rcMvSrchRngRB.setHor(lRangeXRight);
    rcMvSrchRngRB.setVer(lRangeYBottom);
  }
  else
  {
    rcMvSrchRngLT.setHor(cTmpMvPred.getHor() - (iSrchRng << iMvShift));
    rcMvSrchRngLT.setVer(cTmpMvPred.getVer() - (iSrchRng << iMvShift));

    rcMvSrchRngRB.setHor( cTmpMvPred.getHor() + (iSrchRng << iMvShift));
    rcMvSrchRngRB.setVer( cTmpMvPred.getVer() + (iSrchRng << iMvShift) );
  }
#else
  rcMvSrchRngLT.setHor( cTmpMvPred.getHor() - (iSrchRng << iMvShift) );
  rcMvSrchRngLT.setVer( cTmpMvPred.getVer() - (iSrchRng << iMvShift) );

  rcMvSrchRngRB.setHor( cTmpMvPred.getHor() + (iSrchRng << iMvShift));
  rcMvSrchRngRB.setVer( cTmpMvPred.getVer() + (iSrchRng << iMvShift) );
#endif

  pcCU->clipMv        ( rcMvSrchRngLT );
  pcCU->clipMv        ( rcMvSrchRngRB );

#if ME_ENABLE_ROUNDING_OF_MVS
  rcMvSrchRngLT.divideByPowerOf2(iMvShift);
  rcMvSrchRngRB.divideByPowerOf2(iMvShift);
#else
  rcMvSrchRngLT >>= iMvShift;
  rcMvSrchRngRB >>= iMvShift;
#endif
}


Void TEncSearch::xPatternSearch( const TComPattern* const pcPatternKey,
                                 const Pel*               piRefY,
                                 const Int                iRefStride,
                                 const TComMv* const      pcMvSrchRngLT,
                                 const TComMv* const      pcMvSrchRngRB,
                                 TComMv&      rcMv,
                                 Distortion&  ruiSAD )
{
  Int   iSrchRngHorLeft   = pcMvSrchRngLT->getHor();
  Int   iSrchRngHorRight  = pcMvSrchRngRB->getHor();
  Int   iSrchRngVerTop    = pcMvSrchRngLT->getVer();
  Int   iSrchRngVerBottom = pcMvSrchRngRB->getVer();

  Distortion  uiSad;
  Distortion  uiSadBest = std::numeric_limits<Distortion>::max();
  Int         iBestX = 0;
  Int         iBestY = 0;

  //-- jclee for using the SAD function pointer
  m_pcRdCost->setDistParam( pcPatternKey, piRefY, iRefStride,  m_cDistParam );

  // fast encoder decision: use subsampled SAD for integer ME
  if ( m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE1 || m_pcEncCfg->getFastInterSearchMode()==FASTINTERSEARCH_MODE3 )
  {
    if ( m_cDistParam.iRows > 8 )
    {
      m_cDistParam.iSubShift = 1;
    }
  }

  piRefY += (iSrchRngVerTop * iRefStride);
  for ( Int y = iSrchRngVerTop; y <= iSrchRngVerBottom; y++ )
  {
    for ( Int x = iSrchRngHorLeft; x <= iSrchRngHorRight; x++ )
    {
      //  find min. distortion position
      m_cDistParam.pCur = piRefY + x;

      setDistParamComp(COMPONENT_Y);

      m_cDistParam.bitDepth = pcPatternKey->getBitDepthY();
      uiSad = m_cDistParam.DistFunc( &m_cDistParam );

      // motion cost
      uiSad += m_pcRdCost->getCostOfVectorWithPredictor( x, y );

      if ( uiSad < uiSadBest )
      {
        uiSadBest = uiSad;
        iBestX    = x;
        iBestY    = y;
        m_cDistParam.m_maximumDistortionForEarlyExit = uiSad;
      }
    }
    piRefY += iRefStride;
  }

  rcMv.set( iBestX, iBestY );

  ruiSAD = uiSadBest - m_pcRdCost->getCostOfVectorWithPredictor( iBestX, iBestY );
  return;
}


Void TEncSearch::xPatternSearchFast( const TComDataCU* const  pcCU,
                                     const TComPattern* const pcPatternKey,
                                     const Pel* const         piRefY,
                                     const Int                iRefStride,
                                     const TComMv* const      pcMvSrchRngLT,
                                     const TComMv* const      pcMvSrchRngRB,
                                     TComMv&                  rcMv,
                                     Distortion&              ruiSAD,
                                     const TComMv* const      pIntegerMv2Nx2NPred )
{
  assert (MD_LEFT < NUM_MV_PREDICTORS);
  pcCU->getMvPredLeft       ( m_acMvPredictors[MD_LEFT] );
  assert (MD_ABOVE < NUM_MV_PREDICTORS);
  pcCU->getMvPredAbove      ( m_acMvPredictors[MD_ABOVE] );
  assert (MD_ABOVE_RIGHT < NUM_MV_PREDICTORS);
  pcCU->getMvPredAboveRight ( m_acMvPredictors[MD_ABOVE_RIGHT] );

  switch ( m_motionEstimationSearchMethod )
  {
    case MESEARCH_DIAMOND:
      xTZSearch( pcCU, pcPatternKey, piRefY, iRefStride, pcMvSrchRngLT, pcMvSrchRngRB, rcMv, ruiSAD, pIntegerMv2Nx2NPred, false );
      break;

    case MESEARCH_SELECTIVE:
      xTZSearchSelective( pcCU, pcPatternKey, piRefY, iRefStride, pcMvSrchRngLT, pcMvSrchRngRB, rcMv, ruiSAD, pIntegerMv2Nx2NPred );
      break;

    case MESEARCH_DIAMOND_ENHANCED:
      xTZSearch( pcCU, pcPatternKey, piRefY, iRefStride, pcMvSrchRngLT, pcMvSrchRngRB, rcMv, ruiSAD, pIntegerMv2Nx2NPred, true );
      break;

    case MESEARCH_FULL: // shouldn't get here.
    default:
      break;
  }
}


Void TEncSearch::xTZSearch( const TComDataCU* const pcCU,
                            const TComPattern* const pcPatternKey,
                            const Pel* const         piRefY,
                            const Int                iRefStride,
                            const TComMv* const      pcMvSrchRngLT,
                            const TComMv* const      pcMvSrchRngRB,
                            TComMv&                  rcMv,
                            Distortion&              ruiSAD,
                            const TComMv* const      pIntegerMv2Nx2NPred,
                            const Bool               bExtendedSettings)
{
  const Bool bUseAdaptiveRaster                      = bExtendedSettings;
  const Int  iRaster                                 = 5;
  const Bool bTestOtherPredictedMV                   = bExtendedSettings;
  const Bool bTestZeroVector                         = true;
  const Bool bTestZeroVectorStart                    = bExtendedSettings;
  const Bool bTestZeroVectorStop                     = false;
  const Bool bFirstSearchDiamond                     = true;  // 1 = xTZ8PointDiamondSearch   0 = xTZ8PointSquareSearch
  const Bool bFirstCornersForDiamondDist1            = bExtendedSettings;
  const Bool bFirstSearchStop                        = m_pcEncCfg->getFastMEAssumingSmootherMVEnabled();
  const UInt uiFirstSearchRounds                     = 3;     // first search stop X rounds after best match (must be >=1)
  const Bool bEnableRasterSearch                     = true;
  const Bool bAlwaysRasterSearch                     = bExtendedSettings;  // true: BETTER but factor 2 slower
  const Bool bRasterRefinementEnable                 = false; // enable either raster refinement or star refinement
  const Bool bRasterRefinementDiamond                = false; // 1 = xTZ8PointDiamondSearch   0 = xTZ8PointSquareSearch
  const Bool bRasterRefinementCornersForDiamondDist1 = bExtendedSettings;
  const Bool bStarRefinementEnable                   = true;  // enable either star refinement or raster refinement
  const Bool bStarRefinementDiamond                  = true;  // 1 = xTZ8PointDiamondSearch   0 = xTZ8PointSquareSearch
  const Bool bStarRefinementCornersForDiamondDist1   = bExtendedSettings;
  const Bool bStarRefinementStop                     = false;
  const UInt uiStarRefinementRounds                  = 2;  // star refinement stop X rounds after best match (must be >=1)
  const Bool bNewZeroNeighbourhoodTest               = bExtendedSettings;

  UInt uiSearchRange = m_iSearchRange;
  pcCU->clipMv( rcMv );
#if ME_ENABLE_ROUNDING_OF_MVS
  rcMv.divideByPowerOf2(2);
#else
  rcMv >>= 2;
#endif
  // init TZSearchStruct
  IntTZSearchStruct cStruct;
  cStruct.iYStride    = iRefStride;
  cStruct.piRefY      = piRefY;
  cStruct.uiBestSad   = MAX_UINT;

  // set rcMv (Median predictor) as start point and as best point
  xTZSearchHelp( pcPatternKey, cStruct, rcMv.getHor(), rcMv.getVer(), 0, 0 );

  // test whether one of PRED_A, PRED_B, PRED_C MV is better start point than Median predictor
  if ( bTestOtherPredictedMV )
  {
    for ( UInt index = 0; index < NUM_MV_PREDICTORS; index++ )
    {
      TComMv cMv = m_acMvPredictors[index];
      pcCU->clipMv( cMv );
#if ME_ENABLE_ROUNDING_OF_MVS
      cMv.divideByPowerOf2(2);
#else
      cMv >>= 2;
#endif
      if (cMv != rcMv && (cMv.getHor() != cStruct.iBestX && cMv.getVer() != cStruct.iBestY))
      {
        // only test cMV if not obviously previously tested.
        xTZSearchHelp( pcPatternKey, cStruct, cMv.getHor(), cMv.getVer(), 0, 0 );
      }
    }
  }

  // test whether zero Mv is better start point than Median predictor
  if ( bTestZeroVector )
  {
    if ((rcMv.getHor() != 0 || rcMv.getVer() != 0) &&
        (0 != cStruct.iBestX || 0 != cStruct.iBestY))
    {
      // only test 0-vector if not obviously previously tested.
      xTZSearchHelp( pcPatternKey, cStruct, 0, 0, 0, 0 );
    }
  }

  Int   iSrchRngHorLeft   = pcMvSrchRngLT->getHor();
  Int   iSrchRngHorRight  = pcMvSrchRngRB->getHor();
  Int   iSrchRngVerTop    = pcMvSrchRngLT->getVer();
  Int   iSrchRngVerBottom = pcMvSrchRngRB->getVer();

  if (pIntegerMv2Nx2NPred != 0)
  {
    TComMv integerMv2Nx2NPred = *pIntegerMv2Nx2NPred;
    integerMv2Nx2NPred <<= 2;
    pcCU->clipMv( integerMv2Nx2NPred );
#if ME_ENABLE_ROUNDING_OF_MVS
    integerMv2Nx2NPred.divideByPowerOf2(2);
#else
    integerMv2Nx2NPred >>= 2;
#endif
    if ((rcMv != integerMv2Nx2NPred) &&
        (integerMv2Nx2NPred.getHor() != cStruct.iBestX || integerMv2Nx2NPred.getVer() != cStruct.iBestY))
    {
      // only test integerMv2Nx2NPred if not obviously previously tested.
      xTZSearchHelp(pcPatternKey, cStruct, integerMv2Nx2NPred.getHor(), integerMv2Nx2NPred.getVer(), 0, 0);
    }

    // reset search range
    TComMv cMvSrchRngLT;
    TComMv cMvSrchRngRB;
    Int iSrchRng = m_iSearchRange;
    TComMv currBestMv(cStruct.iBestX, cStruct.iBestY );
    currBestMv <<= 2;
#if MCTS_ENC_CHECK
    xSetSearchRange(pcCU, currBestMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB, pcPatternKey);
#else
    xSetSearchRange(pcCU, currBestMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB);
#endif
    iSrchRngHorLeft   = cMvSrchRngLT.getHor();
    iSrchRngHorRight  = cMvSrchRngRB.getHor();
    iSrchRngVerTop    = cMvSrchRngLT.getVer();
    iSrchRngVerBottom = cMvSrchRngRB.getVer();
  }

  if ( m_pcEncCfg->getUseHashBasedME() && pcCU->getPartitionSize( 0 ) == SIZE_2Nx2N )
  {
    TComMv otherMvps[5];
    Int numberOfOtherMvps;
    numberOfOtherMvps = xHashInterPredME( pcCU, pcCU->getWidth( 0 ), pcCU->getHeight( 0 ), m_currRefPicList, m_currRefPicIndex, otherMvps );
    for ( Int i=0; i<numberOfOtherMvps; i++ )
    {
      xTZSearchHelp( pcPatternKey, cStruct, otherMvps[i].getHor(), otherMvps[i].getVer(), 0, 0 );
    }

    if ( numberOfOtherMvps > 0 )
    {
      // write out best match
      rcMv.set( cStruct.iBestX, cStruct.iBestY );
      ruiSAD = cStruct.uiBestSad - m_pcRdCost->getCostOfVectorWithPredictor( cStruct.iBestX, cStruct.iBestY );
      m_bSkipFracME = true;

      return;
    }
  }

  // start search
  Int  iDist = 0;
  Int  iStartX = cStruct.iBestX;
  Int  iStartY = cStruct.iBestY;

  const Bool bBestCandidateZero = (cStruct.iBestX == 0) && (cStruct.iBestY == 0);

  // first search around best position up to now.
  // The following works as a "subsampled/log" window search around the best candidate
  for ( iDist = 1; iDist <= (Int)uiSearchRange; iDist*=2 )
  {
    if ( bFirstSearchDiamond == 1 )
    {
      xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist, bFirstCornersForDiamondDist1 );
    }
    else
    {
      xTZ8PointSquareSearch  ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist );
    }

    if ( bFirstSearchStop && ( cStruct.uiBestRound >= uiFirstSearchRounds ) ) // stop criterion
    {
      break;
    }
  }

  if (!bNewZeroNeighbourhoodTest)
  {
    // test whether zero Mv is a better start point than Median predictor
    if ( bTestZeroVectorStart && ((cStruct.iBestX != 0) || (cStruct.iBestY != 0)) )
    {
      xTZSearchHelp( pcPatternKey, cStruct, 0, 0, 0, 0 );
      if ( (cStruct.iBestX == 0) && (cStruct.iBestY == 0) )
      {
        // test its neighborhood
        for ( iDist = 1; iDist <= (Int)uiSearchRange; iDist*=2 )
        {
          xTZ8PointDiamondSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, 0, 0, iDist, false );
          if ( bTestZeroVectorStop && (cStruct.uiBestRound > 0) ) // stop criterion
          {
            break;
          }
        }
      }
    }
  }
  else
  {
    // Test also zero neighbourhood but with half the range
    // It was reported that the original (above) search scheme using bTestZeroVectorStart did not
    // make sense since one would have already checked the zero candidate earlier
    // and thus the conditions for that test would have not been satisfied
    if (bTestZeroVectorStart == true && bBestCandidateZero != true)
    {
      for ( iDist = 1; iDist <= ((Int)uiSearchRange >> 1); iDist*=2 )
      {
        xTZ8PointDiamondSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, 0, 0, iDist, false );
        if ( bTestZeroVectorStop && (cStruct.uiBestRound > 2) ) // stop criterion
        {
          break;
        }
      }
    }
  }

  // calculate only 2 missing points instead 8 points if cStruct.uiBestDistance == 1
  if ( cStruct.uiBestDistance == 1 )
  {
    cStruct.uiBestDistance = 0;
    xTZ2PointSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB );
  }

  // raster search if distance is too big
  if (bUseAdaptiveRaster)
  {
    Int   iWindowSize = iRaster;
    Int   iSrchRngRasterLeft   = iSrchRngHorLeft;
    Int   iSrchRngRasterRight  = iSrchRngHorRight;
    Int   iSrchRngRasterTop    = iSrchRngVerTop;
    Int   iSrchRngRasterBottom = iSrchRngVerBottom;

    if (!(bEnableRasterSearch && ( ((Int)(cStruct.uiBestDistance) > iRaster))))
    {
      iWindowSize ++;
      iSrchRngRasterLeft /= 2;
      iSrchRngRasterRight /= 2;
      iSrchRngRasterTop /= 2;
      iSrchRngRasterBottom /= 2;
    }
    cStruct.uiBestDistance = iWindowSize;
    for ( iStartY = iSrchRngRasterTop; iStartY <= iSrchRngRasterBottom; iStartY += iWindowSize )
    {
      for ( iStartX = iSrchRngRasterLeft; iStartX <= iSrchRngRasterRight; iStartX += iWindowSize )
      {
        xTZSearchHelp( pcPatternKey, cStruct, iStartX, iStartY, 0, iWindowSize );
      }
    }
  }
  else
  {
    if ( bEnableRasterSearch && ( ((Int)(cStruct.uiBestDistance) > iRaster) || bAlwaysRasterSearch ) )
    {
      cStruct.uiBestDistance = iRaster;
      for ( iStartY = iSrchRngVerTop; iStartY <= iSrchRngVerBottom; iStartY += iRaster )
      {
        for ( iStartX = iSrchRngHorLeft; iStartX <= iSrchRngHorRight; iStartX += iRaster )
        {
          xTZSearchHelp( pcPatternKey, cStruct, iStartX, iStartY, 0, iRaster );
        }
      }
    }
  }

  // raster refinement

  if ( bRasterRefinementEnable && cStruct.uiBestDistance > 0 )
  {
    while ( cStruct.uiBestDistance > 0 )
    {
      iStartX = cStruct.iBestX;
      iStartY = cStruct.iBestY;
      if ( cStruct.uiBestDistance > 1 )
      {
        iDist = cStruct.uiBestDistance >>= 1;
        if ( bRasterRefinementDiamond == 1 )
        {
          xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist, bRasterRefinementCornersForDiamondDist1 );
        }
        else
        {
          xTZ8PointSquareSearch  ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist );
        }
      }

      // calculate only 2 missing points instead 8 points if cStruct.uiBestDistance == 1
      if ( cStruct.uiBestDistance == 1 )
      {
        cStruct.uiBestDistance = 0;
        if ( cStruct.ucPointNr != 0 )
        {
          xTZ2PointSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB );
        }
      }
    }
  }

  // star refinement
  if ( bStarRefinementEnable && cStruct.uiBestDistance > 0 )
  {
    while ( cStruct.uiBestDistance > 0 )
    {
      iStartX = cStruct.iBestX;
      iStartY = cStruct.iBestY;
      cStruct.uiBestDistance = 0;
      cStruct.ucPointNr = 0;
      for ( iDist = 1; iDist < (Int)uiSearchRange + 1; iDist*=2 )
      {
        if ( bStarRefinementDiamond == 1 )
        {
          xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist, bStarRefinementCornersForDiamondDist1 );
        }
        else
        {
          xTZ8PointSquareSearch  ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist );
        }
        if ( bStarRefinementStop && (cStruct.uiBestRound >= uiStarRefinementRounds) ) // stop criterion
        {
          break;
        }
      }

      // calculate only 2 missing points instead 8 points if cStrukt.uiBestDistance == 1
      if ( cStruct.uiBestDistance == 1 )
      {
        cStruct.uiBestDistance = 0;
        if ( cStruct.ucPointNr != 0 )
        {
          xTZ2PointSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB );
        }
      }
    }
  }

  // write out best match
  rcMv.set( cStruct.iBestX, cStruct.iBestY );
  ruiSAD = cStruct.uiBestSad - m_pcRdCost->getCostOfVectorWithPredictor( cStruct.iBestX, cStruct.iBestY );
}


Void TEncSearch::xTZSearchSelective( const TComDataCU* const   pcCU,
                                     const TComPattern* const  pcPatternKey,
                                     const Pel* const          piRefY,
                                     const Int                 iRefStride,
                                     const TComMv* const       pcMvSrchRngLT,
                                     const TComMv* const       pcMvSrchRngRB,
                                     TComMv                   &rcMv,
                                     Distortion               &ruiSAD,
                                     const TComMv* const       pIntegerMv2Nx2NPred )
{
  const Bool bTestOtherPredictedMV    = true;
  const Bool bTestZeroVector          = true;
  const Bool bEnableRasterSearch      = true;
  const Bool bAlwaysRasterSearch      = false;  // 1: BETTER but factor 15x slower
  const Bool bStarRefinementEnable    = true;   // enable either star refinement or raster refinement
  const Bool bStarRefinementDiamond   = true;   // 1 = xTZ8PointDiamondSearch   0 = xTZ8PointSquareSearch
  const Bool bStarRefinementStop      = false;
  const UInt uiStarRefinementRounds   = 2;  // star refinement stop X rounds after best match (must be >=1)
  const UInt uiSearchRange            = m_iSearchRange;
  const Int  uiSearchRangeInitial     = m_iSearchRange >> 2;
  const Int  uiSearchStep             = 4;
  const Int  iMVDistThresh            = 8;

  Int   iSrchRngHorLeft         = pcMvSrchRngLT->getHor();
  Int   iSrchRngHorRight        = pcMvSrchRngRB->getHor();
  Int   iSrchRngVerTop          = pcMvSrchRngLT->getVer();
  Int   iSrchRngVerBottom       = pcMvSrchRngRB->getVer();
  Int   iFirstSrchRngHorLeft    = 0;
  Int   iFirstSrchRngHorRight   = 0;
  Int   iFirstSrchRngVerTop     = 0;
  Int   iFirstSrchRngVerBottom  = 0;
  Int   iStartX                 = 0;
  Int   iStartY                 = 0;
  Int   iBestX                  = 0;
  Int   iBestY                  = 0;
  Int   iDist                   = 0;

  pcCU->clipMv( rcMv );
#if ME_ENABLE_ROUNDING_OF_MVS
  rcMv.divideByPowerOf2(2);
#else
  rcMv >>= 2;
#endif
  // init TZSearchStruct
  IntTZSearchStruct cStruct;
  cStruct.iYStride    = iRefStride;
  cStruct.piRefY      = piRefY;
  cStruct.uiBestSad   = MAX_UINT;
  cStruct.iBestX = 0;
  cStruct.iBestY = 0;


  // set rcMv (Median predictor) as start point and as best point
  xTZSearchHelp( pcPatternKey, cStruct, rcMv.getHor(), rcMv.getVer(), 0, 0 );

  // test whether one of PRED_A, PRED_B, PRED_C MV is better start point than Median predictor
  if ( bTestOtherPredictedMV )
  {
    for ( UInt index = 0; index < NUM_MV_PREDICTORS; index++ )
    {
      TComMv cMv = m_acMvPredictors[index];
      pcCU->clipMv( cMv );
#if ME_ENABLE_ROUNDING_OF_MVS
      cMv.divideByPowerOf2(2);
#else
      cMv >>= 2;
#endif
      xTZSearchHelp( pcPatternKey, cStruct, cMv.getHor(), cMv.getVer(), 0, 0 );
    }
  }

  // test whether zero Mv is better start point than Median predictor
  if ( bTestZeroVector )
  {
    xTZSearchHelp( pcPatternKey, cStruct, 0, 0, 0, 0 );
  }

  if ( pIntegerMv2Nx2NPred != 0 )
  {
    TComMv integerMv2Nx2NPred = *pIntegerMv2Nx2NPred;
    integerMv2Nx2NPred <<= 2;
    pcCU->clipMv( integerMv2Nx2NPred );
#if ME_ENABLE_ROUNDING_OF_MVS
    integerMv2Nx2NPred.divideByPowerOf2(2);
#else
    integerMv2Nx2NPred >>= 2;
#endif
    xTZSearchHelp(pcPatternKey, cStruct, integerMv2Nx2NPred.getHor(), integerMv2Nx2NPred.getVer(), 0, 0);

    // reset search range
    TComMv cMvSrchRngLT;
    TComMv cMvSrchRngRB;
    Int iSrchRng = m_iSearchRange;
    TComMv currBestMv(cStruct.iBestX, cStruct.iBestY );
    currBestMv <<= 2;
#if MCTS_ENC_CHECK
    xSetSearchRange(pcCU, currBestMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB, pcPatternKey);
#else
    xSetSearchRange(pcCU, currBestMv, iSrchRng, cMvSrchRngLT, cMvSrchRngRB);
#endif
    iSrchRngHorLeft   = cMvSrchRngLT.getHor();
    iSrchRngHorRight  = cMvSrchRngRB.getHor();
    iSrchRngVerTop    = cMvSrchRngLT.getVer();
    iSrchRngVerBottom = cMvSrchRngRB.getVer();
  }

  if ( m_pcEncCfg->getUseHashBasedME() && pcCU->getPartitionSize( 0 ) == SIZE_2Nx2N )
  {
    TComMv otherMvps[5];
    Int numberOfOtherMvps;
    numberOfOtherMvps = xHashInterPredME( pcCU, pcCU->getWidth( 0 ), pcCU->getHeight( 0 ), m_currRefPicList, m_currRefPicIndex, otherMvps );
    for ( Int i=0; i<numberOfOtherMvps; i++ )
    {
      xTZSearchHelp( pcPatternKey, cStruct, otherMvps[i].getHor(), otherMvps[i].getVer(), 0, 0 );
    }

    if ( numberOfOtherMvps > 0 )
    {
      // write out best match
      rcMv.set( cStruct.iBestX, cStruct.iBestY );
      ruiSAD = cStruct.uiBestSad - m_pcRdCost->getCostOfVectorWithPredictor( cStruct.iBestX, cStruct.iBestY );
      m_bSkipFracME = true;

      return;
    }
  }
  
  // Initial search
  iBestX = cStruct.iBestX;
  iBestY = cStruct.iBestY; 
  iFirstSrchRngHorLeft    = ((iBestX - uiSearchRangeInitial) > iSrchRngHorLeft)   ? (iBestX - uiSearchRangeInitial) : iSrchRngHorLeft;
  iFirstSrchRngVerTop     = ((iBestY - uiSearchRangeInitial) > iSrchRngVerTop)    ? (iBestY - uiSearchRangeInitial) : iSrchRngVerTop;
  iFirstSrchRngHorRight   = ((iBestX + uiSearchRangeInitial) < iSrchRngHorRight)  ? (iBestX + uiSearchRangeInitial) : iSrchRngHorRight;  
  iFirstSrchRngVerBottom  = ((iBestY + uiSearchRangeInitial) < iSrchRngVerBottom) ? (iBestY + uiSearchRangeInitial) : iSrchRngVerBottom;    

  Bool bFirstVerScan = true;
  for ( iStartY = iFirstSrchRngVerTop; iStartY <= iFirstSrchRngVerBottom; iStartY += uiSearchStep )
  {
    Bool bFirstHorScan = true;
    for ( iStartX = iFirstSrchRngHorLeft; iStartX <= iFirstSrchRngHorRight; iStartX += uiSearchStep )
    {
      xTZSearchHelp( pcPatternKey, cStruct, iStartX, iStartY, 0, 0 );
      xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, 1, false );
      xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, 2, false, !bFirstHorScan, !bFirstVerScan );
      if (bFirstHorScan)
      {
        bFirstHorScan = false;
      }
    }
    if (bFirstVerScan)
    {
      bFirstVerScan = false;
    }
  }

  Int iMaxMVDistToPred = (abs(cStruct.iBestX - iBestX) > iMVDistThresh || abs(cStruct.iBestY - iBestY) > iMVDistThresh);

  //full search with early exit if MV is distant from predictors
  if ( bEnableRasterSearch && (iMaxMVDistToPred || bAlwaysRasterSearch) )
  {
    for ( iStartY = iSrchRngVerTop; iStartY <= iSrchRngVerBottom; iStartY += 1 )
    {
      for ( iStartX = iSrchRngHorLeft; iStartX <= iSrchRngHorRight; iStartX += 1 )
      {
        xTZSearchHelp( pcPatternKey, cStruct, iStartX, iStartY, 0, 1 );
      }
    }
  }
  //Smaller MV, refine around predictor
  else if ( bStarRefinementEnable && cStruct.uiBestDistance > 0 )
  {
    // start refinement
    while ( cStruct.uiBestDistance > 0 )
    {
      iStartX = cStruct.iBestX;
      iStartY = cStruct.iBestY;
      cStruct.uiBestDistance = 0;
      cStruct.ucPointNr = 0;
      for ( iDist = 1; iDist < (Int)uiSearchRange + 1; iDist*=2 )
      {
        if ( bStarRefinementDiamond == 1 )
        {
          xTZ8PointDiamondSearch ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist, false );
        }
        else
        {
          xTZ8PointSquareSearch  ( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB, iStartX, iStartY, iDist );
        }
        if ( bStarRefinementStop && (cStruct.uiBestRound >= uiStarRefinementRounds) ) // stop criterion
        {
          break;
        }
      }

      // calculate only 2 missing points instead 8 points if cStrukt.uiBestDistance == 1
      if ( cStruct.uiBestDistance == 1 )
      {
        cStruct.uiBestDistance = 0;
        if ( cStruct.ucPointNr != 0 )
        {
          xTZ2PointSearch( pcPatternKey, cStruct, pcMvSrchRngLT, pcMvSrchRngRB );
        }
      }
    }
  }

  // write out best match
  rcMv.set( cStruct.iBestX, cStruct.iBestY );
  ruiSAD = cStruct.uiBestSad - m_pcRdCost->getCostOfVectorWithPredictor( cStruct.iBestX, cStruct.iBestY );

}


Void TEncSearch::xPatternSearchFracDIF(
                                       Bool         bIsLosslessCoded,
                                       TComDataCU*  pcCU,
                                       TComPattern* pcPatternKey,
                                       Pel*         piRefY,
                                       Int          iRefStride,
                                       TComMv*      pcMvInt,
                                       TComMv&      rcMvHalf,
                                       TComMv&      rcMvQter,
                                       Distortion&  ruiCost
                                      )
{
  //  Reference pattern initialization (integer scale)
  TComPattern cPatternRoi;
  Int         iOffset    = pcMvInt->getHor() + pcMvInt->getVer() * iRefStride;
  cPatternRoi.initPattern(piRefY + iOffset,
                          pcPatternKey->getROIYWidth(),
                          pcPatternKey->getROIYHeight(),
                          iRefStride,
#if MCTS_ENC_CHECK
                          pcPatternKey->getBitDepthY(),
                          pcPatternKey->getROIYPosX(),
                          pcPatternKey->getROIYPosY());
#else
                          pcPatternKey->getBitDepthY());
#endif
#if MCTS_ENC_CHECK
  cPatternRoi.setTileBorders(pcPatternKey->getTileLeftTopPelPosX(), pcPatternKey->getTileLeftTopPelPosY(), pcPatternKey->getTileRightBottomPelPosX(), pcPatternKey->getTileRightBottomPelPosY());
#endif

  if ( m_bSkipFracME )
  {
    TComMv baseRefMv( 0, 0 );
    rcMvHalf.setZero();
    m_pcRdCost->setCostScale( 0 );
    xExtDIFUpSamplingH( &cPatternRoi );
    rcMvQter = *pcMvInt;   rcMvQter <<= 2;    // for mv-cost
    ruiCost = xPatternRefinement( pcPatternKey, baseRefMv, 1, rcMvQter, !bIsLosslessCoded );
    return;
  }

  //  Half-pel refinement
  xExtDIFUpSamplingH ( &cPatternRoi );

  rcMvHalf = *pcMvInt;   rcMvHalf <<= 1;    // for mv-cost
  TComMv baseRefMv(0, 0);
  ruiCost = xPatternRefinement( pcPatternKey, baseRefMv, 2, rcMvHalf, !bIsLosslessCoded );

  m_pcRdCost->setCostScale( 0 );

  xExtDIFUpSamplingQ ( &cPatternRoi, rcMvHalf );
  baseRefMv = rcMvHalf;
  baseRefMv <<= 1;

  rcMvQter = *pcMvInt;   rcMvQter <<= 1;    // for mv-cost
  rcMvQter += rcMvHalf;  rcMvQter <<= 1;
  ruiCost = xPatternRefinement( pcPatternKey, baseRefMv, 1, rcMvQter, !bIsLosslessCoded );
}


//! encode residual and calculate rate-distortion for a CU block
Void TEncSearch::encodeResAndCalcRdInterCU( TComDataCU* pcCU, TComYuv* pcYuvOrg, TComYuv* pcYuvPred,
                                            TComYuv* pcYuvResi, TComYuv* pcYuvResiBest, TComYuv* pcYuvRec,
                                            Bool bSkipResidual,
                                            TComYuv* pcYuvNoCorrResi,
                                            ACTRDTestTypes eACTRDTestType
                                            DEBUG_STRING_FN_DECLARE(sDebug) )
{
  assert ( !pcCU->isIntra(0) );

  const UInt cuWidthPixels      = pcCU->getWidth ( 0 );
  const UInt cuHeightPixels     = pcCU->getHeight( 0 );
  const Int  numValidComponents = pcCU->getPic()->getNumberValidComponents();
  const TComSPS &sps=*(pcCU->getSlice()->getSPS());

  // The pcCU is not marked as skip-mode at this point, and its m_pcTrCoeff, m_pcArlCoeff, m_puhCbf, m_puhTrIdx will all be 0.
  // due to prior calls to TComDataCU::initEstData(  );
  const Bool iColourTransform = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && ( !pcCU->isLosslessCoded( 0 ) || (sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA )) );
  const Bool extendedPrecision = pcCU->getSlice()->getSPS()->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();
  if ( bSkipResidual ) //  No residual coding : SKIP mode
  {
    pcCU->setSkipFlagSubParts( true, 0, pcCU->getDepth(0) );
    pcCU->setColourTransformSubParts(false, 0, pcCU->getDepth(0));

    pcYuvResi->clear();

    pcYuvPred->copyToPartYuv( pcYuvRec, 0 );
    Distortion distortion = 0;

    for (Int comp=0; comp < numValidComponents; comp++)
    {
      const ComponentID compID=ComponentID(comp);
      const UInt csx=pcYuvOrg->getComponentScaleX(compID);
      const UInt csy=pcYuvOrg->getComponentScaleY(compID);
      distortion += m_pcRdCost->getDistPart( sps.getBitDepth(toChannelType(compID)), pcYuvRec->getAddr(compID), pcYuvRec->getStride(compID), pcYuvOrg->getAddr(compID),
                                               pcYuvOrg->getStride(compID), cuWidthPixels >> csx, cuHeightPixels >> csy, compID);
    }

    m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_CURR_BEST]);
    m_pcEntropyCoder->resetBits();

    if (pcCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
    {
      m_pcEntropyCoder->encodeCUTransquantBypassFlag(pcCU, 0, true);
    }

    m_pcEntropyCoder->encodeSkipFlag(pcCU, 0, true);
    m_pcEntropyCoder->encodeMergeIndex( pcCU, 0, true );

    UInt uiBits = m_pcEntropyCoder->getNumberOfWrittenBits();
    pcCU->getTotalBits()       = uiBits;
    pcCU->getTotalDistortion() = distortion;
    pcCU->getTotalCost()       = m_pcRdCost->calcRdCost( uiBits, distortion );

    m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_TEMP_BEST]);

#if DEBUG_STRING
    pcYuvResiBest->clear(); // Clear the residual image, if we didn't code it.
    for(UInt i=0; i<MAX_NUM_COMPONENT+1; i++)
    {
      sDebug+=debug_reorder_data_token[pcCU->isIntraBC(0)?1:0][i];
    }
#endif

    return;
  }

  //  Residual coding.
   pcCU->setSkipFlagSubParts( false, 0, pcCU->getDepth(0) );
   pcYuvResi->subtract( pcYuvOrg, pcYuvPred, 0, cuWidthPixels );

  TComTURecurse tuLevel0(pcCU, 0);

  Double     nonZeroCost       = 0;
  UInt       nonZeroBits       = 0;
  Distortion nonZeroDistortion = 0;
  Distortion zeroDistortion    = 0;

  if(iColourTransform)
  {
    if(eACTRDTestType == ACT_TWO_CLR || eACTRDTestType == ACT_TRAN_CLR)
    {
      const UInt uiNumSamplesLuma = cuWidthPixels*cuHeightPixels;
      ::memset( m_pTempPel, 0, sizeof( Pel ) * uiNumSamplesLuma );
      zeroDistortion = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA ), m_pTempPel, cuWidthPixels, pcYuvResi->getAddr( COMPONENT_Y, 0 ), pcYuvResi->getStride(COMPONENT_Y), cuWidthPixels, cuHeightPixels, COMPONENT_Y ); // initialized with zero residual destortion
      const UInt csx=pcYuvOrg->getComponentScaleX(COMPONENT_Cb);
      const UInt csy=pcYuvOrg->getComponentScaleY(COMPONENT_Cb);
      zeroDistortion += m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_pTempPel, cuWidthPixels >> csx, pcYuvResi->getAddr( COMPONENT_Cb, 0 ), pcYuvResi->getStride(COMPONENT_Cb), cuWidthPixels >> csx, cuHeightPixels >> csy, COMPONENT_Cb ); // initialized with zero residual destortion
      zeroDistortion += m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_pTempPel, cuWidthPixels >> csx, pcYuvResi->getAddr( COMPONENT_Cr, 0 ), pcYuvResi->getStride(COMPONENT_Cr), cuWidthPixels >> csx, cuHeightPixels >> csy, COMPONENT_Cr ); // initialized with zero residual destortion
    }
    if(eACTRDTestType == ACT_TRAN_CLR)
    {
      pcYuvResi->convert(extendedPrecision, 0, 0, cuWidthPixels, true, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0), pcYuvNoCorrResi);
    }
    else if(eACTRDTestType == ACT_TWO_CLR)
    {
      pcYuvResi->convert(extendedPrecision, 0, 0, cuWidthPixels, true, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0), &(m_pcNoCorrYuvTmp[pcCU->getDepth(0)]));
    }
  }

  m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ pcCU->getDepth( 0 ) ][ CI_CURR_BEST ] );

  if(iColourTransform)
  {
    if(eACTRDTestType == ACT_TWO_CLR)
    {
      xEstimateInterResidualQTTUCSC( pcYuvNoCorrResi, nonZeroCost, nonZeroBits, nonZeroDistortion, tuLevel0, pcYuvResi, eACTRDTestType DEBUG_STRING_PASS_INTO(sDebug) );
    }
    else if(eACTRDTestType == ACT_TRAN_CLR)
    {
      pcCU->setColourTransformSubParts(true, 0 , pcCU->getDepth(0));
      xEstimateInterResidualQT( pcYuvNoCorrResi, nonZeroCost, nonZeroBits, nonZeroDistortion, NULL, tuLevel0 DEBUG_STRING_PASS_INTO(sDebug), pcYuvResi );
    }
    else
    {
      pcCU->setColourTransformSubParts(false, 0, pcCU->getDepth(0));
      xEstimateInterResidualQT( pcYuvResi,  nonZeroCost, nonZeroBits, nonZeroDistortion, &zeroDistortion, tuLevel0 DEBUG_STRING_PASS_INTO(sDebug) );
    }
  }
  else
  {
    pcCU->setColourTransformSubParts(false, 0, pcCU->getDepth(0));
    xEstimateInterResidualQT( pcYuvResi,  nonZeroCost, nonZeroBits, nonZeroDistortion, &zeroDistortion, tuLevel0 DEBUG_STRING_PASS_INTO(sDebug) );
  }

  // -------------------------------------------------------
  // set the coefficients in the pcCU, and also calculates the residual data.
  // If a block full of 0's is efficient, then just use 0's.
  // The costs at this point do not include header bits.

  m_pcEntropyCoder->resetBits();
  m_pcEntropyCoder->encodeQtRootCbfZero( );
  const UInt   zeroResiBits = m_pcEntropyCoder->getNumberOfWrittenBits();
  const Double zeroCost     = (pcCU->isLosslessCoded( 0 )) ? (nonZeroCost+1) : (m_pcRdCost->calcRdCost( zeroResiBits, zeroDistortion ));

  if ( zeroCost < nonZeroCost || !pcCU->getQtRootCbf(0) )
  {
    const UInt uiQPartNum = tuLevel0.GetAbsPartIdxNumParts();
    ::memset( pcCU->getTransformIdx()     , 0, uiQPartNum * sizeof(UChar) );
    ::memset( pcCU->getColourTransform()  , false, uiQPartNum * sizeof(Bool) );
    for (Int comp=0; comp < numValidComponents; comp++)
    {
      const ComponentID component = ComponentID(comp);
      ::memset( pcCU->getCbf( component ) , 0, uiQPartNum * sizeof(UChar) );
      ::memset( pcCU->getCrossComponentPredictionAlpha(component), 0, ( uiQPartNum * sizeof(SChar) ) );
    }
    static const UInt useTS[MAX_NUM_COMPONENT]={0,0,0};
    pcCU->setTransformSkipSubParts ( useTS, 0, pcCU->getDepth(0) );
#if DEBUG_STRING
    sDebug.clear();
    for(UInt i=0; i<MAX_NUM_COMPONENT+1; i++)
    {
      sDebug+=debug_reorder_data_token[pcCU->isIntraBC(0)?1:0][i];
    }
#endif
  }
  else
  {
    xSetInterResidualQTData( NULL, false, tuLevel0); // Call first time to set coefficients.
  }

  // all decisions now made. Fully encode the CU, including the headers:
  m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_CURR_BEST] );

  UInt finalBits = 0;
  xAddSymbolBitsInter( pcCU, finalBits );
  // we've now encoded the pcCU, and so have a valid bit cost

  if ( !pcCU->getQtRootCbf( 0 ) )
  {
    pcYuvResiBest->clear(); // Clear the residual image, if we didn't code it.
  }
  else
  {
    xSetInterResidualQTData( pcYuvResiBest, true, tuLevel0 ); // else set the residual image data pcYUVResiBest from the various temp images.
  }
  m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ pcCU->getDepth( 0 ) ][ CI_TEMP_BEST ] );

  if(iColourTransform && eACTRDTestType == ACT_TRAN_CLR)
  {
    pcYuvResiBest->convert(extendedPrecision, 0, 0, cuWidthPixels, false, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0));
  }

  pcYuvRec->addClip ( pcYuvPred, pcYuvResiBest, 0, cuWidthPixels, sps.getBitDepths() );

  // update with clipped distortion and cost (previously unclipped reconstruction values were used)

  Distortion finalDistortion = 0;
  for(Int comp=0; comp<numValidComponents; comp++)
  {
    const ComponentID compID=ComponentID(comp);
    finalDistortion += m_pcRdCost->getDistPart( sps.getBitDepth(toChannelType(compID)), pcYuvRec->getAddr(compID ), pcYuvRec->getStride(compID ), pcYuvOrg->getAddr(compID ), pcYuvOrg->getStride(compID), cuWidthPixels >> pcYuvOrg->getComponentScaleX(compID), cuHeightPixels >> pcYuvOrg->getComponentScaleY(compID), compID);
  }

  pcCU->getTotalBits()       = finalBits;
  pcCU->getTotalDistortion() = finalDistortion;
  pcCU->getTotalCost()       = m_pcRdCost->calcRdCost( finalBits, finalDistortion );
}



Void TEncSearch::xEstimateInterResidualQT( TComYuv    *pcResi,
                                           Double     &rdCost,
                                           UInt       &ruiBits,
                                           Distortion &ruiDist,
                                           Distortion *puiZeroDist,
                                           TComTU     &rTu
                                           DEBUG_STRING_FN_DECLARE(sDebug),
                                           TComYuv* pcOrgResi
                                          )
{
  TComDataCU *pcCU        = rTu.getCU();
  const UInt uiAbsPartIdx = rTu.GetAbsPartIdxTU();
  const UInt uiDepth      = rTu.GetTransformDepthTotal();
  const UInt uiTrMode     = rTu.GetTransformDepthRel();
  const UInt subTUDepth   = uiTrMode + 1;
  const UInt numValidComp = pcCU->getPic()->getNumberValidComponents();
  const Bool iColourTransform = pcCU->getColourTransform(0);
  const Bool extendedPrecision = pcCU->getSlice()->getSPS()->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();
  DEBUG_STRING_NEW(sSingleStringComp[MAX_NUM_COMPONENT])

  assert( pcCU->getDepth( 0 ) == pcCU->getDepth( uiAbsPartIdx ) );
  const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();

  UInt SplitFlag = ((pcCU->getSlice()->getSPS()->getQuadtreeTUMaxDepthInter() == 1) && pcCU->isInter(uiAbsPartIdx) && ( pcCU->getPartitionSize(uiAbsPartIdx) != SIZE_2Nx2N ));
#if DEBUG_STRING
  const Bool isIntraBc    = pcCU->isIntraBC(uiAbsPartIdx);
  const Int debugPredModeMask = DebugStringGetPredModeMask(pcCU->getPredictionMode(uiAbsPartIdx));
#endif

  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(uiAbsPartIdx) && (pcCU->getWidth(uiAbsPartIdx) >= 32) && (pcCU->isInter(uiAbsPartIdx) || pcCU->isIntraBC(uiAbsPartIdx)) && (pcCU->getPartitionSize(uiAbsPartIdx) != SIZE_2Nx2N))
  {
    SplitFlag = 1;
  }

  Bool bCheckFull;

  if ( SplitFlag && uiDepth == pcCU->getDepth(uiAbsPartIdx) && ( uiLog2TrSize >  pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) ) )
  {
    bCheckFull = false;
  }
  else
  {
    bCheckFull =  ( uiLog2TrSize <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() );
  }

  Bool bCheckSplit  = ( uiLog2TrSize >  pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) );
  if ( m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded( uiAbsPartIdx ) && (pcCU->isIntraBC( uiAbsPartIdx ) || pcCU->isInter( uiAbsPartIdx )) && (pcCU->getWidth( uiAbsPartIdx ) >= 32) && bCheckFull )
  {
    bCheckSplit = false;
  }

  assert( bCheckFull || bCheckSplit );

  // code full block
  Double     dSingleCost = MAX_DOUBLE;
  UInt       uiSingleBits                                                                                                        = 0;
  Distortion uiSingleDistComp            [MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/] = {{0,0},{0,0},{0,0}};
  Distortion uiSingleDist                                                                                                        = 0;
  TCoeff     uiAbsSum                    [MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/] = {{0,0},{0,0},{0,0}};
  UInt       uiBestTransformMode         [MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/] = {{0,0},{0,0},{0,0}};
  //  Stores the best explicit RDPCM mode for a TU encoded without split
  UInt       bestExplicitRdpcmModeUnSplit[MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/] = {{3,3}, {3,3}, {3,3}};
  SChar      bestCrossCPredictionAlpha   [MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/] = {{0,0},{0,0},{0,0}};

  m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );

  if( bCheckFull )
  {
    Double minCost[MAX_NUM_COMPONENT][2/*0 = top (or whole TU for non-4:2:2) sub-TU, 1 = bottom sub-TU*/];
    Bool checkTransformSkip[MAX_NUM_COMPONENT];
    pcCU->setTrIdxSubParts( uiTrMode, uiAbsPartIdx, uiDepth );

    m_pcEntropyCoder->resetBits();

    memset( m_pTempPel, 0, sizeof( Pel ) * rTu.getRect(COMPONENT_Y).width * rTu.getRect(COMPONENT_Y).height ); // not necessary needed for inside of recursion (only at the beginning)

    const UInt uiQTTempAccessLayer = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
    TCoeff *pcCoeffCurr[MAX_NUM_COMPONENT];
#if ADAPTIVE_QP_SELECTION
    TCoeff *pcArlCoeffCurr[MAX_NUM_COMPONENT];
#endif

    for(UInt i=0; i<numValidComp; i++)
    {
      minCost[i][0] = MAX_DOUBLE;
      minCost[i][1] = MAX_DOUBLE;
    }

    Pel crossCPredictedResidualBuffer[ MAX_TU_SIZE * MAX_TU_SIZE ];

    for(UInt i=0; i<numValidComp; i++)
    {
      checkTransformSkip[i]=false;
      const ComponentID compID=ComponentID(i);
      const Int channelBitDepth=pcCU->getSlice()->getSPS()->getBitDepth(toChannelType(compID));
      pcCoeffCurr[compID]    = m_ppcQTTempCoeff[compID][uiQTTempAccessLayer] + rTu.getCoefficientOffset(compID);
#if ADAPTIVE_QP_SELECTION
      pcArlCoeffCurr[compID] = m_ppcQTTempArlCoeff[compID ][uiQTTempAccessLayer] +  rTu.getCoefficientOffset(compID);
#endif

      if(rTu.ProcessComponentSection(compID))
      {
        QpParam cQP(*pcCU, compID, uiAbsPartIdx);
        if(!pcCU->isLosslessCoded(0) && iColourTransform)
        {
          Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
          m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( deltaQP );
          m_pcRdCost->adjustLambdaForColourTrans( deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
        }

        checkTransformSkip[compID] = pcCU->getSlice()->getPPS()->getUseTransformSkip() &&
                                     TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(compID), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize()) &&
                                     (!pcCU->isLosslessCoded(0));

        const Bool splitIntoSubTUs = rTu.getRect(compID).width != rTu.getRect(compID).height;

        TComTURecurse TUIterator(rTu, false, (splitIntoSubTUs ? TComTU::VERTICAL_SPLIT : TComTU::DONT_SPLIT), true, compID);

        const UInt partIdxesPerSubTU = TUIterator.GetAbsPartIdxNumParts(compID);

        do
        {
          const UInt           subTUIndex             = TUIterator.GetSectionNumber();
          const UInt           subTUAbsPartIdx        = TUIterator.GetAbsPartIdxTU(compID);
          const TComRectangle &tuCompRect             = TUIterator.getRect(compID);
          const UInt           subTUBufferOffset      = tuCompRect.width * tuCompRect.height * subTUIndex;

                TCoeff        *currentCoefficients    = pcCoeffCurr[compID] + subTUBufferOffset;
#if ADAPTIVE_QP_SELECTION
                TCoeff        *currentARLCoefficients = pcArlCoeffCurr[compID] + subTUBufferOffset;
#endif
          const Bool isCrossCPredictionAvailable      =    isChroma(compID)
                                                         && pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag()
                                                         && (pcCU->getCbf(subTUAbsPartIdx, COMPONENT_Y, uiTrMode) != 0);

          SChar preCalcAlpha = 0;
          const Pel *pLumaResi = m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix( COMPONENT_Y, rTu.getRect( COMPONENT_Y ).x0, rTu.getRect( COMPONENT_Y ).y0 );

          if (isCrossCPredictionAvailable)
          {
            const Bool bUseReconstructedResidualForEstimate = m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate();
            const Pel  *const lumaResidualForEstimate       = bUseReconstructedResidualForEstimate ? pLumaResi                                                     : pcResi->getAddrPix(COMPONENT_Y, tuCompRect.x0, tuCompRect.y0);
            const UInt        lumaResidualStrideForEstimate = bUseReconstructedResidualForEstimate ? m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(COMPONENT_Y) : pcResi->getStride(COMPONENT_Y);

            preCalcAlpha = xCalcCrossComponentPredictionAlpha(TUIterator,
                                                              compID,
                                                              lumaResidualForEstimate,
                                                              pcResi->getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                              tuCompRect.width,
                                                              tuCompRect.height,
                                                              lumaResidualStrideForEstimate,
                                                              pcResi->getStride(compID));
          }

          const Int transformSkipModesToTest    = checkTransformSkip[compID] ? 2 : 1;
          const Int crossCPredictionModesToTest = (preCalcAlpha != 0)        ? 2 : 1; // preCalcAlpha cannot be anything other than 0 if isCrossCPredictionAvailable is false

          const Bool isOneMode                  = (crossCPredictionModesToTest == 1) && (transformSkipModesToTest == 1);

          for (Int transformSkipModeId = 0; transformSkipModeId < transformSkipModesToTest; transformSkipModeId++)
          {
            pcCU->setTransformSkipPartRange(transformSkipModeId, compID, subTUAbsPartIdx, partIdxesPerSubTU);

            for (Int crossCPredictionModeId = 0; crossCPredictionModeId < crossCPredictionModesToTest; crossCPredictionModeId++)
            {
              const Bool isFirstMode          = (transformSkipModeId == 0) && (crossCPredictionModeId == 0);
              const Bool bUseCrossCPrediction = crossCPredictionModeId != 0;

              m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );
              m_pcEntropyCoder->resetBits();

              pcCU->setTransformSkipPartRange(transformSkipModeId, compID, subTUAbsPartIdx, partIdxesPerSubTU);
              pcCU->setCrossComponentPredictionAlphaPartRange((bUseCrossCPrediction ? preCalcAlpha : 0), compID, subTUAbsPartIdx, partIdxesPerSubTU );

              if ((compID != COMPONENT_Cr) && ((transformSkipModeId == 1) ? m_pcEncCfg->getUseRDOQTS() : m_pcEncCfg->getUseRDOQ()))
              {
                COEFF_SCAN_TYPE scanType = COEFF_SCAN_TYPE(pcCU->getCoefScanIdx(uiAbsPartIdx, tuCompRect.width, tuCompRect.height, compID));
                m_pcEntropyCoder->estimateBit(m_pcTrQuant->m_pcEstBitsSbac, tuCompRect.width, tuCompRect.height, toChannelType(compID), scanType);
              }

#if RDOQ_CHROMA_LAMBDA
              m_pcTrQuant->selectLambda(compID);
#endif

              Pel *pcResiCurrComp = m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
              UInt resiStride     = m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID);

              TCoeff bestCoeffComp   [MAX_TU_SIZE*MAX_TU_SIZE];
              Pel    bestResiComp    [MAX_TU_SIZE*MAX_TU_SIZE];

#if ADAPTIVE_QP_SELECTION
              TCoeff bestArlCoeffComp[MAX_TU_SIZE*MAX_TU_SIZE];
#endif
              TCoeff     currAbsSum   = 0;
              UInt       currCompBits = 0;
              Distortion currCompDist = 0;
              Double     currCompCost = 0;
              UInt       nonCoeffBits = 0;
              Distortion nonCoeffDist = 0;
              Double     nonCoeffCost = 0;

              if(!isOneMode && !isFirstMode)
              {
                memcpy(bestCoeffComp,    currentCoefficients,    (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
                memcpy(bestArlCoeffComp, currentARLCoefficients, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif
                for(Int y = 0; y < tuCompRect.height; y++)
                {
                  memcpy(&bestResiComp[y * tuCompRect.width], (pcResiCurrComp + (y * resiStride)), (sizeof(Pel) * tuCompRect.width));
                }
              }

              if (bUseCrossCPrediction)
              {
                TComTrQuant::crossComponentPrediction(TUIterator,
                                                      compID,
                                                      pLumaResi,
                                                      pcResi->getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                      crossCPredictedResidualBuffer,
                                                      tuCompRect.width,
                                                      tuCompRect.height,
                                                      m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(COMPONENT_Y),
                                                      pcResi->getStride(compID),
                                                      tuCompRect.width,
                                                      false);

                m_pcTrQuant->transformNxN(TUIterator, compID, crossCPredictedResidualBuffer, tuCompRect.width, currentCoefficients,
#if ADAPTIVE_QP_SELECTION
                                          currentARLCoefficients,
#endif
                                          currAbsSum, cQP);
              }
              else
              {
                m_pcTrQuant->transformNxN(TUIterator, compID, pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ), pcResi->getStride(compID), currentCoefficients,
#if ADAPTIVE_QP_SELECTION
                                          currentARLCoefficients,
#endif
                                          currAbsSum, cQP);
              }

              if(isFirstMode || (currAbsSum == 0))
              {
                if (bUseCrossCPrediction)
                {
                  TComTrQuant::crossComponentPrediction(TUIterator,
                                                        compID,
                                                        pLumaResi,
                                                        m_pTempPel,
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                        tuCompRect.width,
                                                        tuCompRect.height,
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(COMPONENT_Y),
                                                        tuCompRect.width,
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID),
                                                        true);

                  nonCoeffDist = m_pcRdCost->getDistPart( channelBitDepth, m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride( compID ), pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          pcResi->getStride(compID), tuCompRect.width, tuCompRect.height, compID); // initialized with zero residual distortion
                }
                else
                {
                  nonCoeffDist = m_pcRdCost->getDistPart( channelBitDepth, m_pTempPel, tuCompRect.width, pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          pcResi->getStride(compID), tuCompRect.width, tuCompRect.height, compID); // initialized with zero residual distortion
                }

                m_pcEntropyCoder->encodeQtCbfZero( TUIterator, toChannelType(compID) );

                if ( isCrossCPredictionAvailable )
                {
                  m_pcEntropyCoder->encodeCrossComponentPrediction( TUIterator, compID );
                }

                nonCoeffBits = m_pcEntropyCoder->getNumberOfWrittenBits();
                nonCoeffCost = m_pcRdCost->calcRdCost( nonCoeffBits, nonCoeffDist );
              }

              if((puiZeroDist != NULL) && isFirstMode)
              {
                *puiZeroDist += nonCoeffDist; // initialized with zero residual distortion
              }

              DEBUG_STRING_NEW(sSingleStringTest)

              if( currAbsSum > 0 ) //if non-zero coefficients are present, a residual needs to be derived for further prediction
              {
                if (isFirstMode)
                {
                  m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );
                  m_pcEntropyCoder->resetBits();
                }

                m_pcEntropyCoder->encodeQtCbf( TUIterator, compID, true );

                if (isCrossCPredictionAvailable)
                {
                  m_pcEntropyCoder->encodeCrossComponentPrediction( TUIterator, compID );
                }

                m_pcEntropyCoder->encodeCoeffNxN( TUIterator, currentCoefficients, compID );
                currCompBits = m_pcEntropyCoder->getNumberOfWrittenBits();

                pcResiCurrComp = m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 );

                m_pcTrQuant->invTransformNxN( TUIterator, compID, pcResiCurrComp, m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID), currentCoefficients, cQP DEBUG_STRING_PASS_INTO_OPTIONAL(&sSingleStringTest, (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask)) );

                if (bUseCrossCPrediction)
                {
                  TComTrQuant::crossComponentPrediction(TUIterator,
                                                        compID,
                                                        pLumaResi,
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                        tuCompRect.width,
                                                        tuCompRect.height,
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(COMPONENT_Y),
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID     ),
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID     ),
                                                        true);
                }

                currCompDist = m_pcRdCost->getDistPart( channelBitDepth, m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                        m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID),
                                                        pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                        pcResi->getStride(compID),
                                                        tuCompRect.width, tuCompRect.height, compID);

                currCompCost = m_pcRdCost->calcRdCost(currCompBits, currCompDist);
                  
                if (pcCU->isLosslessCoded(0))
                {
                  nonCoeffCost = MAX_DOUBLE;
                }
              }
              else if ((transformSkipModeId == 1) && !bUseCrossCPrediction)
              {
                currCompCost = MAX_DOUBLE;
              }
              else
              {
                currCompBits = nonCoeffBits;
                currCompDist = nonCoeffDist;
                currCompCost = nonCoeffCost;
              }

              // evaluate
              if ((currCompCost < minCost[compID][subTUIndex]) || ((transformSkipModeId == 1) && (currCompCost == minCost[compID][subTUIndex])))
              {
                bestExplicitRdpcmModeUnSplit[compID][subTUIndex] = pcCU->getExplicitRdpcmMode(compID, subTUAbsPartIdx);

                if(isFirstMode) //check for forced null
                {
                  if((nonCoeffCost < currCompCost) || (currAbsSum == 0))
                  {
                    memset(currentCoefficients, 0, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));

                    currAbsSum   = 0;
                    currCompBits = nonCoeffBits;
                    currCompDist = nonCoeffDist;
                    currCompCost = nonCoeffCost;
                  }
                }

#if DEBUG_STRING
                if (currAbsSum > 0)
                {
                  DEBUG_STRING_SWAP(sSingleStringComp[compID], sSingleStringTest)
                }
                else
                {
                  sSingleStringComp[compID].clear();
                }
#endif

                uiAbsSum                 [compID][subTUIndex] = currAbsSum;
                uiSingleDistComp         [compID][subTUIndex] = currCompDist;
                minCost                  [compID][subTUIndex] = currCompCost;
                uiBestTransformMode      [compID][subTUIndex] = transformSkipModeId;
                bestCrossCPredictionAlpha[compID][subTUIndex] = (crossCPredictionModeId == 1) ? pcCU->getCrossComponentPredictionAlpha(subTUAbsPartIdx, compID) : 0;

                if (uiAbsSum[compID][subTUIndex] == 0)
                {
                  if (bUseCrossCPrediction)
                  {
                    TComTrQuant::crossComponentPrediction(TUIterator,
                                                          compID,
                                                          pLumaResi,
                                                          m_pTempPel,
                                                          m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                          tuCompRect.width,
                                                          tuCompRect.height,
                                                          m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(COMPONENT_Y),
                                                          tuCompRect.width,
                                                          m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID),
                                                          true);
                  }
                  else
                  {
                    pcResiCurrComp = m_pcQTTempTComYuv[uiQTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
                    const UInt uiStride = m_pcQTTempTComYuv[uiQTTempAccessLayer].getStride(compID);
                    for(UInt uiY = 0; uiY < tuCompRect.height; uiY++)
                    {
                      memset(pcResiCurrComp, 0, (sizeof(Pel) * tuCompRect.width));
                      pcResiCurrComp += uiStride;
                    }
                  }
                }
              }
              else
              {
                // reset
                memcpy(currentCoefficients,    bestCoeffComp,    (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
                memcpy(currentARLCoefficients, bestArlCoeffComp, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif
                for (Int y = 0; y < tuCompRect.height; y++)
                {
                  memcpy((pcResiCurrComp + (y * resiStride)), &bestResiComp[y * tuCompRect.width], (sizeof(Pel) * tuCompRect.width));
                }
              }
            }
          }

          pcCU->setExplicitRdpcmModePartRange            (   bestExplicitRdpcmModeUnSplit[compID][subTUIndex],                            compID, subTUAbsPartIdx, partIdxesPerSubTU);
          pcCU->setTransformSkipPartRange                (   uiBestTransformMode         [compID][subTUIndex],                            compID, subTUAbsPartIdx, partIdxesPerSubTU );
          pcCU->setCbfPartRange                          ((((uiAbsSum                    [compID][subTUIndex] > 0) ? 1 : 0) << uiTrMode), compID, subTUAbsPartIdx, partIdxesPerSubTU );
          pcCU->setCrossComponentPredictionAlphaPartRange(   bestCrossCPredictionAlpha   [compID][subTUIndex],                            compID, subTUAbsPartIdx, partIdxesPerSubTU );
        } while (TUIterator.nextSection(rTu)); //end of sub-TU loop

        if(!pcCU->isLosslessCoded(0) && iColourTransform)
        {
          Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
          m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( - deltaQP );
          m_pcRdCost->adjustLambdaForColourTrans( - deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
        }
      } // processing section
    } // component loop

    {
      if(iColourTransform)
      {
        const TComRectangle &tuCompRect=rTu.getRect(COMPONENT_Y);
        for(UInt ch = 0; ch < numValidComp; ch++)
        {
          const ComponentID compID=ComponentID(ch);
          const TComRectangle &tuCompRectTmp = rTu.getRect(compID);
          assert(tuCompRectTmp.width == tuCompRectTmp.height);

          m_pcQTTempTComYuv[uiQTTempAccessLayer].copyPartToPartComponentMxN(compID, &m_tmpYuvPred, tuCompRectTmp);
        }
        m_tmpYuvPred.convert( extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, false, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(uiAbsPartIdx) );

        uiSingleDistComp[COMPONENT_Y ][0] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA ), m_tmpYuvPred.getAddrPix( COMPONENT_Y, tuCompRect.x0, tuCompRect.y0 ), m_tmpYuvPred.getStride(COMPONENT_Y),
          pcOrgResi->getAddrPix( COMPONENT_Y, tuCompRect.x0, tuCompRect.y0 ), pcOrgResi->getStride(COMPONENT_Y), tuCompRect.width, tuCompRect.height, COMPONENT_Y );

        const TComRectangle &tuCompRectC=rTu.getRect(COMPONENT_Cb);
        uiSingleDistComp[COMPONENT_Cb][0] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_tmpYuvPred.getAddrPix( COMPONENT_Cb, tuCompRectC.x0, tuCompRectC.y0 ), m_tmpYuvPred.getStride(COMPONENT_Cb),
          pcOrgResi->getAddrPix( COMPONENT_Cb, tuCompRectC.x0, tuCompRectC.y0 ), pcOrgResi->getStride(COMPONENT_Cb), tuCompRectC.width, tuCompRectC.height, COMPONENT_Cb );

        uiSingleDistComp[COMPONENT_Cr][0] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_tmpYuvPred.getAddrPix( COMPONENT_Cr, tuCompRectC.x0, tuCompRectC.y0 ), m_tmpYuvPred.getStride(COMPONENT_Cr),
          pcOrgResi->getAddrPix( COMPONENT_Cr, tuCompRectC.x0, tuCompRectC.y0 ), pcOrgResi->getStride(COMPONENT_Cr), tuCompRectC.width, tuCompRectC.height, COMPONENT_Cr );

        uiSingleDistComp[COMPONENT_Y][1] = uiSingleDistComp[COMPONENT_Cb][1] = uiSingleDistComp[COMPONENT_Cr][1] = 0;
      }
    }

    if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && (pcCU->getCbf(uiAbsPartIdx, COMPONENT_Y, uiTrMode) || pcCU->getCbf(uiAbsPartIdx, COMPONENT_Cb, uiTrMode) || pcCU->getCbf(uiAbsPartIdx, COMPONENT_Cr, uiTrMode)) )
    {
      m_pcEntropyCoder->m_pcEntropyCoderIf->codeColourTransformFlag( pcCU, uiAbsPartIdx );
    }

    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const ComponentID compID = ComponentID(ch);
      if (rTu.ProcessComponentSection(compID) && (rTu.getRect(compID).width != rTu.getRect(compID).height))
      {
        offsetSubTUCBFs(rTu, compID); //the CBFs up to now have been defined for two sub-TUs - shift them down a level and replace with the parent level CBF
      }
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );
    m_pcEntropyCoder->resetBits();

    if( uiLog2TrSize > pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) )
    {
      m_pcEntropyCoder->encodeTransformSubdivFlag( 0, 5 - uiLog2TrSize );
    }

    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const UInt chOrderChange = ((ch + 1) == numValidComp) ? 0 : (ch + 1);
      const ComponentID compID=ComponentID(chOrderChange);
      if( rTu.ProcessComponentSection(compID) )
      {
        m_pcEntropyCoder->encodeQtCbf( rTu, compID, true );
      }
    }

    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const ComponentID compID=ComponentID(ch);
      if (rTu.ProcessComponentSection(compID))
      {
        if(isChroma(compID) && (uiAbsSum[COMPONENT_Y][0] != 0))
        {
          m_pcEntropyCoder->encodeCrossComponentPrediction( rTu, compID );
        }

        m_pcEntropyCoder->encodeCoeffNxN( rTu, pcCoeffCurr[compID], compID );
        for (UInt subTUIndex = 0; subTUIndex < 2; subTUIndex++)
        {
          uiSingleDist += uiSingleDistComp[compID][subTUIndex];
        }
      }
    }

    uiSingleBits = m_pcEntropyCoder->getNumberOfWrittenBits();

    dSingleCost = m_pcRdCost->calcRdCost( uiSingleBits, uiSingleDist );
  } // check full

  // code sub-blocks
  if( bCheckSplit )
  {
    if( bCheckFull )
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_TEST ] );
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );
    }
    Distortion uiSubdivDist = 0;
    UInt       uiSubdivBits = 0;
    Double     dSubdivCost = 0.0;

    //save the non-split CBFs in case we need to restore them later

    UInt bestCBF     [MAX_NUM_COMPONENT];
    UInt bestsubTUCBF[MAX_NUM_COMPONENT][2];
    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const ComponentID compID=ComponentID(ch);

      if (rTu.ProcessComponentSection(compID))
      {
        bestCBF[compID] = pcCU->getCbf(uiAbsPartIdx, compID, uiTrMode);

        const TComRectangle &tuCompRect = rTu.getRect(compID);
        if (tuCompRect.width != tuCompRect.height)
        {
          const UInt partIdxesPerSubTU = rTu.GetAbsPartIdxNumParts(compID) >> 1;

          for (UInt subTU = 0; subTU < 2; subTU++)
          {
            bestsubTUCBF[compID][subTU] = pcCU->getCbf ((uiAbsPartIdx + (subTU * partIdxesPerSubTU)), compID, subTUDepth);
          }
        }
      }
    }


    TComTURecurse tuRecurseChild(rTu, false);
    const UInt uiQPartNumSubdiv = tuRecurseChild.GetAbsPartIdxNumParts();

    DEBUG_STRING_NEW(sSplitString[MAX_NUM_COMPONENT])

    do
    {
      DEBUG_STRING_NEW(childString)
      xEstimateInterResidualQT( pcResi, dSubdivCost, uiSubdivBits, uiSubdivDist, bCheckFull ? NULL : puiZeroDist,  tuRecurseChild DEBUG_STRING_PASS_INTO(childString), pcOrgResi );
#if DEBUG_STRING
      // split the string by component and append to the relevant output (because decoder decodes in channel order, whereas this search searches by TU-order)
      std::size_t lastPos=0;
      const std::size_t endStrng=childString.find(debug_reorder_data_token[isIntraBc?1:0][MAX_NUM_COMPONENT], lastPos);
      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        if (lastPos!=std::string::npos && childString.find(debug_reorder_data_inter_token[ch], lastPos)==lastPos)
        {
          lastPos+=strlen(debug_reorder_data_inter_token[ch]); // skip leading string
        }
        if (pos!=std::string::npos && pos>endStrng)
        {
          lastPos=endStrng;
        }
        sSplitString[ch]+=childString.substr(lastPos, (pos==std::string::npos)? std::string::npos : (pos-lastPos) );
        lastPos=pos;
      }
#endif
    } while ( tuRecurseChild.nextSection(rTu) ) ;

    UInt uiCbfAny=0;
    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      UInt uiYUVCbf = 0;
      for( UInt ui = 0; ui < 4; ++ui )
      {
        uiYUVCbf |= pcCU->getCbf( uiAbsPartIdx + ui * uiQPartNumSubdiv, ComponentID(ch),  uiTrMode + 1 );
      }
      UChar *pBase=pcCU->getCbf( ComponentID(ch) );
      const UInt flags=uiYUVCbf << uiTrMode;
      for( UInt ui = 0; ui < 4 * uiQPartNumSubdiv; ++ui )
      {
        pBase[uiAbsPartIdx + ui] |= flags;
      }
      uiCbfAny|=uiYUVCbf;
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_ROOT ] );
    m_pcEntropyCoder->resetBits();

    // when compID isn't a channel, code Cbfs:
    xEncodeInterResidualQT( MAX_NUM_COMPONENT, rTu );
    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      xEncodeInterResidualQT( ComponentID(ch), rTu );
    }

    uiSubdivBits = m_pcEntropyCoder->getNumberOfWrittenBits();
    dSubdivCost  = m_pcRdCost->calcRdCost( uiSubdivBits, uiSubdivDist );

    if (!bCheckFull || (uiCbfAny && (dSubdivCost < dSingleCost)))
    {
      rdCost += dSubdivCost;
      ruiBits += uiSubdivBits;
      ruiDist += uiSubdivDist;
#if DEBUG_STRING
      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        DEBUG_STRING_APPEND(sDebug, debug_reorder_data_token[isIntraBc?1:0][ch])
        DEBUG_STRING_APPEND(sDebug, sSplitString[ch])
      }
#endif
    }
    else
    {
      rdCost  += dSingleCost;
      ruiBits += uiSingleBits;
      ruiDist += uiSingleDist;

      //restore state to unsplit

      pcCU->setTrIdxSubParts( uiTrMode, uiAbsPartIdx, uiDepth );

      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        const ComponentID compID=ComponentID(ch);

        DEBUG_STRING_APPEND(sDebug, debug_reorder_data_token[isIntraBc?1:0][ch])
        if (rTu.ProcessComponentSection(compID))
        {
          DEBUG_STRING_APPEND(sDebug, sSingleStringComp[compID])

          const Bool splitIntoSubTUs   = rTu.getRect(compID).width != rTu.getRect(compID).height;
          const UInt numberOfSections  = splitIntoSubTUs ? 2 : 1;
          const UInt partIdxesPerSubTU = rTu.GetAbsPartIdxNumParts(compID) >> (splitIntoSubTUs ? 1 : 0);

          for (UInt subTUIndex = 0; subTUIndex < numberOfSections; subTUIndex++)
          {
            const UInt  uisubTUPartIdx = uiAbsPartIdx + (subTUIndex * partIdxesPerSubTU);

            if (splitIntoSubTUs)
            {
              const UChar combinedCBF = (bestsubTUCBF[compID][subTUIndex] << subTUDepth) | (bestCBF[compID] << uiTrMode);
              pcCU->setCbfPartRange(combinedCBF, compID, uisubTUPartIdx, partIdxesPerSubTU);
            }
            else
            {
              pcCU->setCbfPartRange((bestCBF[compID] << uiTrMode), compID, uisubTUPartIdx, partIdxesPerSubTU);
            }

            pcCU->setCrossComponentPredictionAlphaPartRange(bestCrossCPredictionAlpha[compID][subTUIndex], compID, uisubTUPartIdx, partIdxesPerSubTU);
            pcCU->setTransformSkipPartRange(uiBestTransformMode[compID][subTUIndex], compID, uisubTUPartIdx, partIdxesPerSubTU);
            pcCU->setExplicitRdpcmModePartRange(bestExplicitRdpcmModeUnSplit[compID][subTUIndex], compID, uisubTUPartIdx, partIdxesPerSubTU);
          }
        }
      }

      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ uiDepth ][ CI_QT_TRAFO_TEST ] );
    }
  }
  else
  {
    rdCost  += dSingleCost;
    ruiBits += uiSingleBits;
    ruiDist += uiSingleDist;
#if DEBUG_STRING
    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const ComponentID compID=ComponentID(ch);
      DEBUG_STRING_APPEND(sDebug, debug_reorder_data_token[isIntraBc?1:0][compID])

      if (rTu.ProcessComponentSection(compID))
      {
        DEBUG_STRING_APPEND(sDebug, sSingleStringComp[compID])
      }
    }
#endif
  }
  DEBUG_STRING_APPEND(sDebug, debug_reorder_data_token[isIntraBc?1:0][MAX_NUM_COMPONENT])
}



Void TEncSearch::xEncodeInterResidualQT( const ComponentID compID, TComTU &rTu )
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt uiAbsPartIdx=rTu.GetAbsPartIdxTU();
  const UInt uiCurrTrMode = rTu.GetTransformDepthRel();
  assert( pcCU->getDepth( 0 ) == pcCU->getDepth( uiAbsPartIdx ) );
  const UInt uiTrMode = pcCU->getTransformIdx( uiAbsPartIdx );

  const Bool bSubdiv = uiCurrTrMode != uiTrMode;

  const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();

  if (compID==MAX_NUM_COMPONENT)  // we are not processing a channel, instead we always recurse and code the CBFs
  {
    if( uiLog2TrSize <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() && uiLog2TrSize > pcCU->getQuadtreeTULog2MinSizeInCU(uiAbsPartIdx) )
    {
      if((pcCU->getSlice()->getSPS()->getQuadtreeTUMaxDepthInter() == 1) && (pcCU->getPartitionSize(uiAbsPartIdx) != SIZE_2Nx2N))
      {
        assert(bSubdiv); // Inferred splitting rule - see derivation and use of interSplitFlag in the specification.
      }
      else
      {
        m_pcEntropyCoder->encodeTransformSubdivFlag( bSubdiv, 5 - uiLog2TrSize );
      }
    }

    assert( !pcCU->isIntra(uiAbsPartIdx) );

    const Bool bFirstCbfOfCU = uiCurrTrMode == 0;

    for (UInt ch=COMPONENT_Cb; ch<pcCU->getPic()->getNumberValidComponents(); ch++)
    {
      const ComponentID compIdInner=ComponentID(ch);
      if( bFirstCbfOfCU || rTu.ProcessingAllQuadrants(compIdInner) )
      {
        if( bFirstCbfOfCU || pcCU->getCbf( uiAbsPartIdx, compIdInner, uiCurrTrMode - 1 ) )
        {
          m_pcEntropyCoder->encodeQtCbf( rTu, compIdInner, !bSubdiv );
        }
      }
      else
      {
        assert( pcCU->getCbf( uiAbsPartIdx, compIdInner, uiCurrTrMode ) == pcCU->getCbf( uiAbsPartIdx, compIdInner, uiCurrTrMode - 1 ) );
      }
    }

    if (!bSubdiv)
    {
      m_pcEntropyCoder->encodeQtCbf( rTu, COMPONENT_Y, true );
      if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && (pcCU->getCbf(uiAbsPartIdx, COMPONENT_Y, uiTrMode) || pcCU->getCbf(uiAbsPartIdx, COMPONENT_Cb, uiTrMode) || pcCU->getCbf(uiAbsPartIdx, COMPONENT_Cr, uiTrMode)) )
      {
        m_pcEntropyCoder->m_pcEntropyCoderIf->codeColourTransformFlag( pcCU, uiAbsPartIdx );
      }
    }
  }

  if( !bSubdiv )
  {
    if (compID != MAX_NUM_COMPONENT) // we have already coded the CBFs, so now we code coefficients
    {
      if (rTu.ProcessComponentSection(compID))
      {
        if (isChroma(compID) && (pcCU->getCbf(uiAbsPartIdx, COMPONENT_Y, uiTrMode) != 0))
        {
          m_pcEntropyCoder->encodeCrossComponentPrediction(rTu, compID);
        }

        if (pcCU->getCbf(uiAbsPartIdx, compID, uiTrMode) != 0)
        {
          const UInt uiQTTempAccessLayer = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
          TCoeff *pcCoeffCurr = m_ppcQTTempCoeff[compID][uiQTTempAccessLayer] + rTu.getCoefficientOffset(compID);
          m_pcEntropyCoder->encodeCoeffNxN( rTu, pcCoeffCurr, compID );
        }
      }
    }
  }
  else
  {
    if( compID==MAX_NUM_COMPONENT || pcCU->getCbf( uiAbsPartIdx, compID, uiCurrTrMode ) )
    {
      TComTURecurse tuRecurseChild(rTu, false);
      do
      {
        xEncodeInterResidualQT( compID, tuRecurseChild );
      } while (tuRecurseChild.nextSection(rTu));
    }
  }
}




Void TEncSearch::xSetInterResidualQTData( TComYuv* pcResi, Bool bSpatial, TComTU &rTu ) // TODO: turn this into two functions for bSpatial=true and false.
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt uiCurrTrMode=rTu.GetTransformDepthRel();
  const UInt uiAbsPartIdx=rTu.GetAbsPartIdxTU();
  assert( pcCU->getDepth( 0 ) == pcCU->getDepth( uiAbsPartIdx ) );
  const UInt uiTrMode = pcCU->getTransformIdx( uiAbsPartIdx );
  const TComSPS *sps=pcCU->getSlice()->getSPS();

  if( uiCurrTrMode == uiTrMode )
  {
    const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();
    const UInt uiQTTempAccessLayer = sps->getQuadtreeTULog2MaxSize() - uiLog2TrSize;

    if( bSpatial )
    {
      // Data to be copied is in the spatial domain, i.e., inverse-transformed.

      for(UInt i=0; i<pcResi->getNumberValidComponents(); i++)
      {
        const ComponentID compID=ComponentID(i);
        if (rTu.ProcessComponentSection(compID))
        {
          const TComRectangle &rectCompTU(rTu.getRect(compID));
          m_pcQTTempTComYuv[uiQTTempAccessLayer].copyPartToPartComponentMxN    ( compID, pcResi, rectCompTU );
        }
      }
    }
    else
    {
      for (UInt ch=0; ch < getNumberValidComponents(sps->getChromaFormatIdc()); ch++)
      {
        const ComponentID compID   = ComponentID(ch);
        if (rTu.ProcessComponentSection(compID))
        {
          const TComRectangle &rectCompTU(rTu.getRect(compID));
          const UInt numCoeffInBlock    = rectCompTU.width * rectCompTU.height;
          const UInt offset             = rTu.getCoefficientOffset(compID);
          TCoeff* dest                  = pcCU->getCoeff(compID)                        + offset;
          const TCoeff* src             = m_ppcQTTempCoeff[compID][uiQTTempAccessLayer] + offset;
          ::memcpy( dest, src, sizeof(TCoeff)*numCoeffInBlock );

#if ADAPTIVE_QP_SELECTION
          TCoeff* pcArlCoeffSrc            = m_ppcQTTempArlCoeff[compID][uiQTTempAccessLayer] + offset;
          TCoeff* pcArlCoeffDst            = pcCU->getArlCoeff(compID)                        + offset;
          ::memcpy( pcArlCoeffDst, pcArlCoeffSrc, sizeof( TCoeff ) * numCoeffInBlock );
#endif
        }
      }
    }
  }
  else
  {

    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xSetInterResidualQTData( pcResi, bSpatial, tuRecurseChild );
    } while (tuRecurseChild.nextSection(rTu));
  }
}




UInt TEncSearch::xModeBitsIntra( TComDataCU* pcCU, UInt uiMode, UInt uiPartOffset, UInt uiDepth, const ChannelType chType )
{
  // Reload only contexts required for coding intra mode information
  m_pcRDGoOnSbacCoder->loadIntraDirMode( m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST], chType );

  // Temporarily set the intra dir being tested, and only
  // for absPartIdx, since encodeIntraDirModeLuma/Chroma only use
  // the entry at absPartIdx.

  UChar &rIntraDirVal=pcCU->getIntraDir( chType )[uiPartOffset];
  UChar origVal=rIntraDirVal;
  rIntraDirVal = uiMode;
  //pcCU->setIntraDirSubParts ( chType, uiMode, uiPartOffset, uiDepth + uiInitTrDepth );

  m_pcEntropyCoder->resetBits();
  if (isLuma(chType))
  {
    m_pcEntropyCoder->encodeIntraDirModeLuma ( pcCU, uiPartOffset);
  }
  else
  {
    m_pcEntropyCoder->encodeIntraDirModeChroma ( pcCU, uiPartOffset);
  }

  rIntraDirVal = origVal; // restore

  return m_pcEntropyCoder->getNumberOfWrittenBits();
}




UInt TEncSearch::xUpdateCandList( UInt uiMode, Double uiCost, UInt uiFastCandNum, UInt * CandModeList, Double * CandCostList )
{
  UInt i;
  UInt shift=0;

  while ( shift<uiFastCandNum && uiCost<CandCostList[ uiFastCandNum-1-shift ] )
  {
    shift++;
  }

  if( shift!=0 )
  {
    for(i=1; i<shift; i++)
    {
      CandModeList[ uiFastCandNum-i ] = CandModeList[ uiFastCandNum-1-i ];
      CandCostList[ uiFastCandNum-i ] = CandCostList[ uiFastCandNum-1-i ];
    }
    CandModeList[ uiFastCandNum-shift ] = uiMode;
    CandCostList[ uiFastCandNum-shift ] = uiCost;
    return 1;
  }

  return 0;
}





/** add inter-prediction syntax elements for a CU block
 * \param pcCU
 * \param uiQp
 * \param uiTrMode
 * \param ruiBits
 * \returns Void
 */
Void  TEncSearch::xAddSymbolBitsInter( TComDataCU* pcCU, UInt& ruiBits )
{
  if(pcCU->getMergeFlag( 0 ) && pcCU->getPartitionSize( 0 ) == SIZE_2Nx2N && !pcCU->getQtRootCbf( 0 ))
  {
    pcCU->setSkipFlagSubParts( true, 0, pcCU->getDepth(0) );

    m_pcEntropyCoder->resetBits();
    if(pcCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
    {
      m_pcEntropyCoder->encodeCUTransquantBypassFlag(pcCU, 0, true);
    }
    m_pcEntropyCoder->encodeSkipFlag(pcCU, 0, true);
    m_pcEntropyCoder->encodeMergeIndex(pcCU, 0, true);

    ruiBits += m_pcEntropyCoder->getNumberOfWrittenBits();
  }
  else
  {
    m_pcEntropyCoder->resetBits();

    if(pcCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
    {
      m_pcEntropyCoder->encodeCUTransquantBypassFlag(pcCU, 0, true);
    }

    m_pcEntropyCoder->encodeSkipFlag ( pcCU, 0, true );
    m_pcEntropyCoder->encodePredMode( pcCU, 0, true );
    Bool codeDeltaQp = false;
    Bool codeChromaQpAdj = false;
    m_pcEntropyCoder->encodePaletteModeInfo( pcCU, 0, true, &codeDeltaQp, &codeChromaQpAdj );    
    m_pcEntropyCoder->encodePartSize( pcCU, 0, pcCU->getDepth(0), true );
    m_pcEntropyCoder->encodePredInfo( pcCU, 0 );

    m_pcEntropyCoder->encodeCoeff   ( pcCU, 0, pcCU->getDepth(0), codeDeltaQp, codeChromaQpAdj );

    ruiBits += m_pcEntropyCoder->getNumberOfWrittenBits();
  }
}





/**
 * \brief Generate half-sample interpolated block
 *
 * \param pattern Reference picture ROI
 * \param biPred    Flag indicating whether block is for biprediction
 */
Void TEncSearch::xExtDIFUpSamplingH( TComPattern* pattern )
{
  Int width      = pattern->getROIYWidth();
  Int height     = pattern->getROIYHeight();
  Int srcStride  = pattern->getPatternLStride();

  Int intStride = m_filteredBlockTmp[0].getStride(COMPONENT_Y);
  Int dstStride = m_filteredBlock[0][0].getStride(COMPONENT_Y);
  Pel *intPtr;
  Pel *dstPtr;
  Int filterSize = NTAPS_LUMA;
  Int halfFilterSize = (filterSize>>1);
  Pel *srcPtr = pattern->getROIY() - halfFilterSize*srcStride - 1;

  const ChromaFormat chFmt = m_filteredBlock[0][0].getChromaFormat();

  m_if.filterHor(COMPONENT_Y, srcPtr, srcStride, m_filteredBlockTmp[0].getAddr(COMPONENT_Y), intStride, width+1, height+filterSize, 0, false, chFmt, pattern->getBitDepthY());
  if ( !m_bSkipFracME )
  {
    m_if.filterHor(COMPONENT_Y, srcPtr, srcStride, m_filteredBlockTmp[2].getAddr(COMPONENT_Y), intStride, width+1, height+filterSize, 2, false, chFmt, pattern->getBitDepthY());
  }


  intPtr = m_filteredBlockTmp[0].getAddr(COMPONENT_Y) + halfFilterSize * intStride + 1;
  dstPtr = m_filteredBlock[0][0].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width+0, height+0, 0, false, true, chFmt, pattern->getBitDepthY());

  if ( m_bSkipFracME )
  {
    return;
  }

  intPtr = m_filteredBlockTmp[0].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride + 1;
  dstPtr = m_filteredBlock[2][0].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width+0, height+1, 2, false, true, chFmt, pattern->getBitDepthY());

  intPtr = m_filteredBlockTmp[2].getAddr(COMPONENT_Y) + halfFilterSize * intStride;
  dstPtr = m_filteredBlock[0][2].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width+1, height+0, 0, false, true, chFmt, pattern->getBitDepthY());

  intPtr = m_filteredBlockTmp[2].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
  dstPtr = m_filteredBlock[2][2].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width+1, height+1, 2, false, true, chFmt, pattern->getBitDepthY());
}





/**
 * \brief Generate quarter-sample interpolated blocks
 *
 * \param pattern    Reference picture ROI
 * \param halfPelRef Half-pel mv
 * \param biPred     Flag indicating whether block is for biprediction
 */
Void TEncSearch::xExtDIFUpSamplingQ( TComPattern* pattern, TComMv halfPelRef )
{
  Int width      = pattern->getROIYWidth();
  Int height     = pattern->getROIYHeight();
  Int srcStride  = pattern->getPatternLStride();

  Pel *srcPtr;
  Int intStride = m_filteredBlockTmp[0].getStride(COMPONENT_Y);
  Int dstStride = m_filteredBlock[0][0].getStride(COMPONENT_Y);
  Pel *intPtr;
  Pel *dstPtr;
  Int filterSize = NTAPS_LUMA;

  Int halfFilterSize = (filterSize>>1);

  Int extHeight = (halfPelRef.getVer() == 0) ? height + filterSize : height + filterSize-1;

  const ChromaFormat chFmt = m_filteredBlock[0][0].getChromaFormat();

  // Horizontal filter 1/4
  srcPtr = pattern->getROIY() - halfFilterSize * srcStride - 1;
  intPtr = m_filteredBlockTmp[1].getAddr(COMPONENT_Y);
  if (halfPelRef.getVer() > 0)
  {
    srcPtr += srcStride;
  }
  if (halfPelRef.getHor() >= 0)
  {
    srcPtr += 1;
  }
  m_if.filterHor(COMPONENT_Y, srcPtr, srcStride, intPtr, intStride, width, extHeight, 1, false, chFmt, pattern->getBitDepthY());

  // Horizontal filter 3/4
  srcPtr = pattern->getROIY() - halfFilterSize*srcStride - 1;
  intPtr = m_filteredBlockTmp[3].getAddr(COMPONENT_Y);
  if (halfPelRef.getVer() > 0)
  {
    srcPtr += srcStride;
  }
  if (halfPelRef.getHor() > 0)
  {
    srcPtr += 1;
  }
  m_if.filterHor(COMPONENT_Y, srcPtr, srcStride, intPtr, intStride, width, extHeight, 3, false, chFmt, pattern->getBitDepthY());

  // Generate @ 1,1
  intPtr = m_filteredBlockTmp[1].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
  dstPtr = m_filteredBlock[1][1].getAddr(COMPONENT_Y);
  if (halfPelRef.getVer() == 0)
  {
    intPtr += intStride;
  }
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 1, false, true, chFmt, pattern->getBitDepthY());

  // Generate @ 3,1
  intPtr = m_filteredBlockTmp[1].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
  dstPtr = m_filteredBlock[3][1].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 3, false, true, chFmt, pattern->getBitDepthY());

  if (halfPelRef.getVer() != 0)
  {
    // Generate @ 2,1
    intPtr = m_filteredBlockTmp[1].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
    dstPtr = m_filteredBlock[2][1].getAddr(COMPONENT_Y);
    if (halfPelRef.getVer() == 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 2, false, true, chFmt, pattern->getBitDepthY());

    // Generate @ 2,3
    intPtr = m_filteredBlockTmp[3].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
    dstPtr = m_filteredBlock[2][3].getAddr(COMPONENT_Y);
    if (halfPelRef.getVer() == 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 2, false, true, chFmt, pattern->getBitDepthY());
  }
  else
  {
    // Generate @ 0,1
    intPtr = m_filteredBlockTmp[1].getAddr(COMPONENT_Y) + halfFilterSize * intStride;
    dstPtr = m_filteredBlock[0][1].getAddr(COMPONENT_Y);
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 0, false, true, chFmt, pattern->getBitDepthY());

    // Generate @ 0,3
    intPtr = m_filteredBlockTmp[3].getAddr(COMPONENT_Y) + halfFilterSize * intStride;
    dstPtr = m_filteredBlock[0][3].getAddr(COMPONENT_Y);
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 0, false, true, chFmt, pattern->getBitDepthY());
  }

  if (halfPelRef.getHor() != 0)
  {
    // Generate @ 1,2
    intPtr = m_filteredBlockTmp[2].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
    dstPtr = m_filteredBlock[1][2].getAddr(COMPONENT_Y);
    if (halfPelRef.getHor() > 0)
    {
      intPtr += 1;
    }
    if (halfPelRef.getVer() >= 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 1, false, true, chFmt, pattern->getBitDepthY());

    // Generate @ 3,2
    intPtr = m_filteredBlockTmp[2].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
    dstPtr = m_filteredBlock[3][2].getAddr(COMPONENT_Y);
    if (halfPelRef.getHor() > 0)
    {
      intPtr += 1;
    }
    if (halfPelRef.getVer() > 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 3, false, true, chFmt, pattern->getBitDepthY());
  }
  else
  {
    // Generate @ 1,0
    intPtr = m_filteredBlockTmp[0].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride + 1;
    dstPtr = m_filteredBlock[1][0].getAddr(COMPONENT_Y);
    if (halfPelRef.getVer() >= 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 1, false, true, chFmt, pattern->getBitDepthY());

    // Generate @ 3,0
    intPtr = m_filteredBlockTmp[0].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride + 1;
    dstPtr = m_filteredBlock[3][0].getAddr(COMPONENT_Y);
    if (halfPelRef.getVer() > 0)
    {
      intPtr += intStride;
    }
    m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 3, false, true, chFmt, pattern->getBitDepthY());
  }

  // Generate @ 1,3
  intPtr = m_filteredBlockTmp[3].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
  dstPtr = m_filteredBlock[1][3].getAddr(COMPONENT_Y);
  if (halfPelRef.getVer() == 0)
  {
    intPtr += intStride;
  }
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 1, false, true, chFmt, pattern->getBitDepthY());

  // Generate @ 3,3
  intPtr = m_filteredBlockTmp[3].getAddr(COMPONENT_Y) + (halfFilterSize-1) * intStride;
  dstPtr = m_filteredBlock[3][3].getAddr(COMPONENT_Y);
  m_if.filterVer(COMPONENT_Y, intPtr, intStride, dstPtr, dstStride, width, height, 3, false, true, chFmt, pattern->getBitDepthY());
}





//! set wp tables
Void  TEncSearch::setWpScalingDistParam( TComDataCU* pcCU, Int iRefIdx, RefPicList eRefPicListCur )
{
  if ( iRefIdx<0 )
  {
    m_cDistParam.bApplyWeight = false;
    return;
  }

  TComSlice       *pcSlice  = pcCU->getSlice();
  WPScalingParam  *wp0 , *wp1;

  m_cDistParam.bApplyWeight = ( pcSlice->getSliceType()==P_SLICE && pcSlice->testWeightPred() ) || ( pcSlice->getSliceType()==B_SLICE && pcSlice->testWeightBiPred() ) ;

  if ( !m_cDistParam.bApplyWeight )
  {
    return;
  }

  Int iRefIdx0 = ( eRefPicListCur == REF_PIC_LIST_0 ) ? iRefIdx : (-1);
  Int iRefIdx1 = ( eRefPicListCur == REF_PIC_LIST_1 ) ? iRefIdx : (-1);

  getWpScaling( pcCU, iRefIdx0, iRefIdx1, wp0 , wp1 );

  if ( iRefIdx0 < 0 )
  {
    wp0 = NULL;
  }
  if ( iRefIdx1 < 0 )
  {
    wp1 = NULL;
  }

  m_cDistParam.wpCur  = NULL;

  if ( eRefPicListCur == REF_PIC_LIST_0 )
  {
    m_cDistParam.wpCur = wp0;
  }
  else
  {
    m_cDistParam.wpCur = wp1;
  }
}

// SCM new added functions
Void TEncSearch::xEncColorTransformFlagQT( TComTU &rTu )
{
  TComDataCU* pcCU=rTu.getCU();
  const UInt absPartIdx = rTu.GetAbsPartIdxTU();
  const UInt trDepth=rTu.GetTransformDepthRel();

  const UInt  trMode        = pcCU->getTransformIdx( absPartIdx );
  const UInt  subdiv        = ( trMode > trDepth ? 1 : 0 );

  if( subdiv )
  {
    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xEncColorTransformFlagQT( tuRecurseChild );
    } while (tuRecurseChild.nextSection(rTu) );
  }
  else
  {
    //===== color transform flag =====
    Bool hasCodedCoeff = (pcCU->getCbf(absPartIdx, COMPONENT_Y,  trDepth) || pcCU->getCbf(absPartIdx, COMPONENT_Cb, trDepth) || pcCU->getCbf(absPartIdx, COMPONENT_Cr, trDepth));
    if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && pcCU->hasAssociatedACTFlag(absPartIdx) && hasCodedCoeff )
    {
      m_pcEntropyCoder->m_pcEntropyCoderIf->codeColourTransformFlag( pcCU, absPartIdx );
    }
  }
}

Void
TEncSearch::xIntraCodingTUBlockCSC(       TComYuv*    pcResiYuv,
                                          Pel         reconResiLuma[MAX_CU_SIZE * MAX_CU_SIZE],
                                          const Bool        checkCrossCPrediction,
                                          const ComponentID compID,
                                          TComTU&     rTu,
                                          QpParam&    cQP
                                          DEBUG_STRING_FN_DECLARE(sDebug)
                                  )
{
  assert( rTu.ProcessComponentSection(compID) );

  TComDataCU*         pcCU              = rTu.getCU();
  const TComRectangle &rect             = rTu.getRect(compID);
  const UInt          log2TrSize        = rTu.GetLog2LumaTrSize();
  const UInt          absPartIdx        = rTu.GetAbsPartIdxTU();
  const UInt          QTLayer           = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - log2TrSize;
  const ChromaFormat  chFmt             = rTu.GetChromaFormat();
  const ChannelType   chType            = toChannelType(compID);
  assert( chFmt == CHROMA_444 );

  const UInt          width             = rect.width;
  const UInt          height            = rect.height;
  Pel*                piResi            = pcResiYuv->getAddr( compID, absPartIdx );
  Pel*                piRecQt           = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
  Pel                 codedResi[MAX_CU_SIZE * MAX_CU_SIZE];
  const UInt          stride            = pcResiYuv->getStride (compID);
  const UInt          recQtStride       = m_pcQTTempTComYuv[ QTLayer ].getStride(compID);
  const UInt          codedResiStride   = MAX_CU_SIZE;

  const UInt          chPredMode         = pcCU->getIntraDir( chType, absPartIdx );
  const Bool          bUseCrossCPrediction = isChroma(compID) && (chPredMode == DM_CHROMA_IDX) && checkCrossCPrediction;
  const Int           blkX                 = g_auiRasterToPelX[ g_auiZscanToRaster[ absPartIdx ] ];
  const Int           blkY                 = g_auiRasterToPelY[ g_auiZscanToRaster[ absPartIdx ] ];
  const Int           bufferOffset         = blkX + (blkY * MAX_CU_SIZE);
  Pel  *const reconstructedLumaResidual    = reconResiLuma + bufferOffset;

#if DEBUG_STRING
  const Int debugPredModeMask=DebugStringGetPredModeMask(MODE_INTRA);
#endif

  TCoeff*       pcCoeff           = m_ppcQTTempCoeff[compID][QTLayer] + rTu.getCoefficientOffset(compID);
#if ADAPTIVE_QP_SELECTION
  TCoeff*       pcArlCoeff        = m_ppcQTTempArlCoeff[compID][ QTLayer ] + rTu.getCoefficientOffset(compID);
#endif
  Bool          useTransformSkip  = pcCU->getTransformSkip(absPartIdx, compID);

  // get residual
  Pel*  pcCodedResi = codedResi;
  Pel*  pcResi      = piResi;

  for( UInt y = 0; y < height; y++ )
  {
    for ( UInt x = 0; x < width; x++ )
    {
      pcCodedResi[x] = pcResi[x];
    }

    pcResi      += stride;
    pcCodedResi += codedResiStride;
  }

  if ( pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag() && bUseCrossCPrediction )
  {
    TComTrQuant::crossComponentPrediction( rTu, compID, reconstructedLumaResidual, codedResi, codedResi, width, height, MAX_CU_SIZE, codedResiStride, codedResiStride, false );
  }

  //--- transform and quantization ---
  if (useTransformSkip ? m_pcEncCfg->getUseRDOQTS() : m_pcEncCfg->getUseRDOQ())
  {
    COEFF_SCAN_TYPE scanType = COEFF_SCAN_TYPE(pcCU->getCoefScanIdx(absPartIdx, width, height, compID));
    m_pcEntropyCoder->estimateBit(m_pcTrQuant->m_pcEstBitsSbac, width, height, chType, scanType);
  }

  //--- transform and quantization ---
  TCoeff uiAbsSum = 0;

#if RDOQ_CHROMA_LAMBDA
  m_pcTrQuant->selectLambda     (compID);
#endif

  m_pcTrQuant->transformNxN     ( rTu, compID, codedResi, codedResiStride, pcCoeff,
#if ADAPTIVE_QP_SELECTION
    pcArlCoeff,
#endif
    uiAbsSum, cQP
  );

  //--- inverse transform ---
  DEBUG_STRING_NEW(sTemp)

#if DEBUG_STRING
  if ( (uiAbsSum > 0) || (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask) )
#else
  if ( uiAbsSum > 0 )
#endif
  {
    pcCodedResi = codedResi;
    m_pcTrQuant->invTransformNxN ( rTu, compID, pcCodedResi, codedResiStride, pcCoeff, cQP DEBUG_STRING_PASS_INTO_OPTIONAL(&sDebug, (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask))  );
  }
  else
  {
    pcCodedResi = codedResi;
    memset( pcCoeff, 0, sizeof( TCoeff ) * width * height );
    for( UInt y = 0; y < height; y++ )
    {
      memset( pcCodedResi, 0, sizeof( Pel ) * width );
      pcCodedResi += codedResiStride;
    }
  }

  //--- residue reconstruction ---
  if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
  {
    if ( bUseCrossCPrediction )
    {
      TComTrQuant::crossComponentPrediction( rTu, compID, reconstructedLumaResidual, codedResi, codedResi, width, height, MAX_CU_SIZE, codedResiStride, codedResiStride, true );
    }
    else if ( isLuma( compID ) )
    {
      xStoreCrossComponentPredictionResult( reconstructedLumaResidual, codedResi, rTu, 0, 0, MAX_CU_SIZE, codedResiStride );
    }
  }

  Pel* pCodedResi   = codedResi;
  Pel* pRecQt       = piRecQt;

  for( UInt y = 0; y < height; y++ )
  {
    for ( UInt x = 0; x < width; x++ )
    {
      pRecQt[x] = pCodedResi[x];  //recon rsidue
    }

    pCodedResi += codedResiStride;
    pRecQt     += recQtStride;
  }
}


Void
TEncSearch::xRecurIntraCodingQTTUCSC( TComYuv* pcOrgYuv, TComYuv* pcPredYuv, TComYuv* pcResiYuv, Distortion& PUDistY, Distortion& PUDistC, Double& dPUCost, TComTU& rTu, Bool bTestMaxTUSize, ACTRDTestTypes eACTRDTestType DEBUG_STRING_FN_DECLARE(sDebug) )
{
  TComDataCU          *pcCU                 = rTu.getCU();
  const UInt          trDepth               = rTu.GetTransformDepthRel();
  const UInt          fullDepth             = rTu.GetTransformDepthTotal();
  const UInt          absPartIdx            = rTu.GetAbsPartIdxTU();
  const UInt          partIndxNumPerTU      = rTu.GetAbsPartIdxNumParts(COMPONENT_Y);
  const ChromaFormat  chFmt                 = rTu.GetChromaFormat();
  const UInt          log2TrSize            = rTu.GetLog2LumaTrSize();
  const UInt          QTLayer               = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - log2TrSize;
  const UInt          numberValidComponents = getNumberValidComponents(chFmt);
  Bool                bCheckFull            = ( log2TrSize  <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() );
  Bool                bCheckSplit           = ( bTestMaxTUSize && bCheckFull )? false: ( log2TrSize  >  pcCU->getQuadtreeTULog2MinSizeInCU(absPartIdx) );
  const Bool          extendedPrecision     = rTu.getCU()->getSlice()->getSPS()->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();;

  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(absPartIdx) && bCheckFull)
  {
    bCheckSplit = false;
  }

  assert( bCheckFull || bCheckSplit );
  assert( chFmt == CHROMA_444);

  Double              dSingleCost                                 = MAX_DOUBLE;
  Distortion          singleDist[MAX_NUM_CHANNEL_TYPE]            = {0, 0};

  UInt                singleColorSpaceId                          = 0;
  Double              dSingleColorSpaceCost[2]                        = {MAX_DOUBLE, MAX_DOUBLE};
  Distortion          singleColorSpaceDist[2][MAX_NUM_CHANNEL_TYPE]   = {{0, 0}, {0, 0}};
  UInt                singleColorSpaceBits[2]                         = {0, 0};

  Double              dSingleComponentCost[2][MAX_NUM_COMPONENT]              = {{MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE}, {MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE}};
  UInt                singleComponentCbf[2][MAX_NUM_COMPONENT]                = {{0, 0, 0}, {0, 0, 0}};
  UInt                singleComponentTransformMode[2][MAX_NUM_COMPONENT]      = {{0, 0, 0}, {0, 0, 0}};
  SChar               cSingleComponentPredictionAlpha[2][MAX_NUM_COMPONENT]   = {{0, 0, 0}, {0, 0, 0}};

  UInt relTrDepth = trDepth;
  if( pcCU->getPartitionSize(absPartIdx) == SIZE_NxN )
  {
    relTrDepth --;
  }
  m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled  = MAX_DOUBLE;
  m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled = MAX_DOUBLE;
  m_sACTRDCostTU[relTrDepth].uiIsCSCEnabled       = 2;

  if( bCheckFull )
  {
    m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );

    pcCU->setTrIdxSubParts( trDepth, absPartIdx, fullDepth );

    //intra prediction
    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      const ComponentID   compID    = ComponentID(ch);
      const ChannelType   chType    = toChannelType(compID);
      const TComRectangle &rect     = rTu.getRect(compID);
      const UInt          width     = rect.width;
      const UInt          height    = rect.height;

      const Bool bIsLuma            = isLuma(compID);
      const UInt chPredMode         = pcCU->getIntraDir( chType, absPartIdx );
      const UInt chFinalMode        = (chPredMode == DM_CHROMA_IDX && !bIsLuma) ? pcCU->getIntraDir(CHANNEL_TYPE_LUMA, absPartIdx) : chPredMode;

      const Bool bUseFilteredPredictions = TComPrediction::filteringIntraReferenceSamples(compID, chFinalMode, width, height, chFmt, pcCU->getSlice()->getSPS()->getSpsRangeExtension().getIntraSmoothingDisabledFlag());
      Pel*       piOrg                   = pcOrgYuv ->getAddr( compID, absPartIdx );
      Pel*       piPred                  = pcPredYuv->getAddr( compID, absPartIdx );
      Pel*       piResi                  = pcResiYuv->getAddr( compID, absPartIdx );
      const UInt stride                  = pcOrgYuv ->getStride (compID);

      DEBUG_STRING_NEW(sTemp)
      initIntraPatternChType( rTu, compID, bUseFilteredPredictions DEBUG_STRING_PASS_INTO(sTemp) );
      predIntraAng( compID, chFinalMode, piOrg, stride, piPred, stride, rTu, bUseFilteredPredictions );

      Pel* pcOrg  = piOrg;
      Pel* pcPred = piPred;
      Pel* pcResi = piResi;

      for( UInt y = 0; y < height; y++ )
      {
        for( UInt x = 0; x < width; x++ )
        {
          pcResi[ x ] = pcOrg[ x ] - pcPred[ x ];
        }

        pcOrg  += stride;
        pcResi += stride;
        pcPred += stride;
      }
    }

    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      const ComponentID compID = ComponentID(ch);
      pcResiYuv->copyPartToPartComponent( compID, &(m_pcQTTempTComYuvCS[QTLayer]), absPartIdx, rTu.getRect(compID).width, rTu.getRect(compID).height );
    }

    for(Int colorSpaceId = 0; colorSpaceId < 2; colorSpaceId++)
    {
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );

      Bool bCheckACT = m_pcEncCfg->getRGBFormatFlag()? (!colorSpaceId? true: false): (colorSpaceId? true: false);
      assert( pcCU->getIntraDir( CHANNEL_TYPE_CHROMA, absPartIdx ) == DM_CHROMA_IDX );

      if(eACTRDTestType == ACT_TRAN_CLR && !bCheckACT)
      {
        continue;
      }
      if(eACTRDTestType == ACT_ORG_CLR && bCheckACT)
      {
        continue;
      }

      pcCU->setColourTransformSubParts(bCheckACT, absPartIdx, fullDepth);
      if( colorSpaceId )
      {
        for( UInt ch = 0; ch < numberValidComponents; ch++ )
        {
          const ComponentID compID = ComponentID(ch);
          m_pcQTTempTComYuvCS[QTLayer].copyPartToPartComponent( compID, pcResiYuv, absPartIdx, rTu.getRect(compID).width, rTu.getRect(compID).height );
        }
      }

      if(bCheckACT)
      {
        pcResiYuv->convert(extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, true, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0));
      }

      SChar preCalcAlpha[MAX_NUM_COMPONENT] = {0, 0, 0};

      if( pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag() && !m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate() )
      {
        for( UInt ch = 0; ch < numberValidComponents; ch++ )
        {
          const ComponentID    compID                        = ComponentID(ch);
          const TComRectangle &tuCompRect                    = rTu.getRect(compID);
          const Pel  *const    lumaResidualForEstimate       = pcResiYuv->getAddr( COMPONENT_Y, absPartIdx );
          const UInt           lumaResidualStrideForEstimate = pcResiYuv->getStride ( COMPONENT_Y );

          const UInt           chPredMode                  = pcCU->getIntraDir( toChannelType(compID), absPartIdx );
          const Bool bDecorrelationAvailable                 = isChroma(compID) && (chPredMode == DM_CHROMA_IDX);
          if( bDecorrelationAvailable )
          {
            preCalcAlpha[compID] = xCalcCrossComponentPredictionAlpha(rTu,
                                                                      compID,
                                                                      lumaResidualForEstimate,
                                                                      pcResiYuv->getAddr( compID, absPartIdx ),
                                                                      tuCompRect.width,
                                                                      tuCompRect.height,
                                                                      lumaResidualStrideForEstimate,
                                                                      pcResiYuv->getStride( compID ) );
          }
        }
      }

      Pel reconResiLuma   [MAX_CU_SIZE * MAX_CU_SIZE];
      Pel reconResiLumaTmp[MAX_CU_SIZE * MAX_CU_SIZE];

      for( UInt ch = 0; ch < numberValidComponents; ch++ )
      {
        Double     dSingleCostTmp  = MAX_DOUBLE;
        UInt       singleBitsTmp = 0;
        Distortion singleDistTmp = 0;
        UInt       singleCbfTmp  = 0;

        const ComponentID    compID     = ComponentID(ch);
        const TComRectangle &tuCompRect = rTu.getRect(compID);
        QpParam cQP(*pcCU, compID, absPartIdx);

        if( !pcCU->isLosslessCoded(0) && bCheckACT )
        {
          Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
          m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( deltaQP );
          m_pcRdCost->adjustLambdaForColourTrans( deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
        }

        Bool bCheckTransformSkip = pcCU->getSlice()->getPPS()->getUseTransformSkip() &&
                                   TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(compID), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize()) &&
                                   (!pcCU->isLosslessCoded(0));
        if( m_pcEncCfg->getUseTransformSkipFast() )
        {
          bCheckTransformSkip &= (pcCU->getPartitionSize(absPartIdx) == SIZE_NxN);
          if(isChroma(compID))
          {
            bCheckTransformSkip &= (pcCU->getTransformSkip(absPartIdx, COMPONENT_Y) != 0);
          }
        }

        Bool bCheckCrossComponentPrediction =    isChroma(compID)
                                              && pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag()
                                              && (pcCU->getIntraDir( toChannelType(compID), absPartIdx ) == DM_CHROMA_IDX)
                                              && (pcCU->getCbf(absPartIdx, COMPONENT_Y, trDepth) != 0);

        SChar cCalcAlpha = 0;
        if ( bCheckCrossComponentPrediction && m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate() )
        {
          cCalcAlpha = xCalcCrossComponentPredictionAlpha(rTu,
                                                          compID,
                                                          reconResiLuma + tuCompRect.y0 * MAX_CU_SIZE + tuCompRect.x0,
                                                          pcResiYuv->getAddr( compID, absPartIdx ),
                                                          tuCompRect.width,
                                                          tuCompRect.height,
                                                          MAX_CU_SIZE,
                                                          pcResiYuv->getStride( compID ) );
        }
        else if( bCheckCrossComponentPrediction )
        {
          cCalcAlpha = preCalcAlpha[compID];
        }
        bCheckCrossComponentPrediction &= (cCalcAlpha != 0);

        m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );

        const Int transformSkipModesToTest    = bCheckTransformSkip               ? 2 : 1;
        const Int crossCPredictionModesToTest = bCheckCrossComponentPrediction    ? 2 : 1;

        for(Int transformSkipModeId = 0; transformSkipModeId < transformSkipModesToTest; transformSkipModeId++)
        {
          for(Int crossCPredictionModeId = 0; crossCPredictionModeId < crossCPredictionModesToTest; crossCPredictionModeId++)
          {
            DEBUG_STRING_NEW(sModeString)
            m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );
            m_pcEntropyCoder->resetBits();

            pcCU->setTransformSkipPartRange                ( transformSkipModeId,                          compID, absPartIdx, partIndxNumPerTU );
            pcCU->setCrossComponentPredictionAlphaPartRange( (crossCPredictionModeId != 0)? cCalcAlpha: 0, compID, absPartIdx, partIndxNumPerTU );

            xIntraCodingTUBlockCSC( pcResiYuv, (compID == COMPONENT_Y)? reconResiLumaTmp: reconResiLuma, (crossCPredictionModeId != 0), compID, rTu, cQP DEBUG_STRING_PASS_INTO(sModeString) );
            singleCbfTmp = pcCU->getCbf( absPartIdx, compID, trDepth );

            if( (singleCbfTmp == 0) && (transformSkipModeId != 0) )
            {
              dSingleCostTmp = MAX_DOUBLE;
            }
            else
            {
              if( compID == COMPONENT_Y )
              {
                singleBitsTmp = xGetIntraBitsQT( rTu, true, false, false );
              }
              else
              {
                singleBitsTmp = xGetIntraBitsQTChroma( rTu, compID, false );
              }

              //residual distortion in YCgCo domain
              singleDistTmp = m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType( compID )],
                                                        m_pcQTTempTComYuv[QTLayer].getAddr(compID, absPartIdx),
                                                        m_pcQTTempTComYuv[QTLayer].getStride(compID),
                                                        pcResiYuv->getAddr(compID, absPartIdx),
                                                        pcResiYuv->getStride(compID),
                                                        rTu.getRect(compID).width,
                                                        rTu.getRect(compID).height,
                                                        compID);
              dSingleCostTmp = m_pcRdCost->calcRdCost( singleBitsTmp, singleDistTmp );
            }

            if( dSingleCostTmp < dSingleComponentCost[colorSpaceId][compID] )
            {
              dSingleComponentCost[colorSpaceId][compID]             = dSingleCostTmp;
              singleComponentCbf[colorSpaceId][compID]             = singleCbfTmp;
              cSingleComponentPredictionAlpha[colorSpaceId][compID]  = (crossCPredictionModeId != 0) ? pcCU->getCrossComponentPredictionAlpha(absPartIdx, compID) : 0;
              singleComponentTransformMode[colorSpaceId][compID]   = transformSkipModeId;

              xStoreIntraResultQT(compID, rTu);

              m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_TEMP_BEST ] );

              if( compID == COMPONENT_Y )
              {
                const Int xOffset = rTu.getRect( COMPONENT_Y ).x0;
                const Int yOffset = rTu.getRect( COMPONENT_Y ).y0;
                xStoreCrossComponentPredictionResult(reconResiLuma, reconResiLumaTmp, rTu, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE);
              }
            }
          }
        }

        pcCU->setTransformSkipPartRange(singleComponentTransformMode[colorSpaceId][compID],                      compID, absPartIdx, partIndxNumPerTU );
        pcCU->setCbfPartRange          ((singleComponentCbf[colorSpaceId][compID] << trDepth),                 compID, absPartIdx, partIndxNumPerTU );
        pcCU->setCrossComponentPredictionAlphaPartRange(cSingleComponentPredictionAlpha[colorSpaceId][compID],     compID, absPartIdx, partIndxNumPerTU );
        xLoadIntraResultQT(compID, rTu );

        m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_TEMP_BEST ] );

        if( !pcCU->isLosslessCoded(0) && bCheckACT )
        {
          Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
          m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( - deltaQP );
          m_pcRdCost->adjustLambdaForColourTrans            ( - deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
        }
      }

      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
      m_pcEntropyCoder->resetBits();
      singleColorSpaceBits[colorSpaceId] = xGetIntraBitsQT( rTu, true, true, false );

      if( bCheckACT )
      {
        m_pcQTTempTComYuv[ QTLayer ].convert(extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, false, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0));
      }

      for( UInt ch = 0; ch < numberValidComponents; ch++ )
      {
        ComponentID         compID           = ComponentID(ch);
        const TComRectangle &rect            = rTu.getRect(compID);
        const UInt          width            = rect.width;
        const UInt          height           = rect.height;
        Pel*                pcPred           = pcPredYuv->getAddr( compID, absPartIdx );
        Pel*                pcResi           = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
        Pel*                pcRecQt          = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
        const UInt          stride           = pcPredYuv->getStride (compID);
        const UInt          recQtStride      = m_pcQTTempTComYuv[ QTLayer ].getStride(compID);
        const UInt          clipbd           = pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(compID)];

        for( UInt y = 0; y < height; y++ )
        {
          for( UInt x = 0; x < width; x++ )
          {
            pcRecQt[ x ] = Pel( ClipBD<Int>( Int(pcPred[x]) + Int(pcResi[x]), clipbd ) );
          }

          pcPred     += stride;
          pcResi     += recQtStride;
          pcRecQt    += recQtStride;
        }
      }

      singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_LUMA] = m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(COMPONENT_Y)],
                                                                                       m_pcQTTempTComYuv[QTLayer].getAddr(COMPONENT_Y, absPartIdx),
                                                                                       m_pcQTTempTComYuv[QTLayer].getStride(COMPONENT_Y),
                                                                                       pcOrgYuv->getAddr(COMPONENT_Y, absPartIdx),
                                                                                       pcOrgYuv->getStride(COMPONENT_Y),
                                                                                       rTu.getRect(COMPONENT_Y).width,
                                                                                       rTu.getRect(COMPONENT_Y).height,
                                                                                       COMPONENT_Y);
      singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_CHROMA] = 0;
      for(UInt ch = COMPONENT_Cb; ch < numberValidComponents; ch++)
      {
        ComponentID compID = ComponentID(ch);
        singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_CHROMA] += m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(compID)],
                                                                                            m_pcQTTempTComYuv[QTLayer].getAddr(compID, absPartIdx),
                                                                                            m_pcQTTempTComYuv[QTLayer].getStride(compID),
                                                                                            pcOrgYuv->getAddr(compID, absPartIdx),
                                                                                            pcOrgYuv->getStride(compID),
                                                                                            rTu.getRect(compID).width,
                                                                                            rTu.getRect(compID).height,
                                                                                            compID);
      }

      dSingleColorSpaceCost[colorSpaceId] = m_pcRdCost->calcRdCost( singleColorSpaceBits[colorSpaceId], singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_LUMA] + singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_CHROMA] );

      if(colorSpaceId)
      {
        m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled = dSingleColorSpaceCost[colorSpaceId];
      }
      else
      {
        m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled  = dSingleColorSpaceCost[colorSpaceId];
      }

      if( dSingleColorSpaceCost[colorSpaceId] < dSingleCost )
      {
        dSingleCost                       = dSingleColorSpaceCost[colorSpaceId];
        singleDist[CHANNEL_TYPE_LUMA]   = singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_LUMA];
        singleDist[CHANNEL_TYPE_CHROMA] = singleColorSpaceDist[colorSpaceId][CHANNEL_TYPE_CHROMA];
        singleColorSpaceId              = colorSpaceId;

        xStoreIntraResultQT(COMPONENT_Y,  rTu, true);
        xStoreIntraResultQT(COMPONENT_Cb, rTu, true);
        xStoreIntraResultQT(COMPONENT_Cr, rTu, true);
        m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_CHROMA_INTRA ] );  //CI_CHROMA_INTRA is never used
      }

      if( !colorSpaceId && !singleComponentCbf[colorSpaceId][COMPONENT_Y] && !singleComponentCbf[colorSpaceId][COMPONENT_Cb] && !singleComponentCbf[colorSpaceId][COMPONENT_Cr] )  //all cbfs are zero, skip the other color space
      {
        break;
      }
      if( !colorSpaceId && relTrDepth && m_sACTRDCostTU[relTrDepth-1].uiIsCSCEnabled == 1 )
      {
        break;
      }
    }

    Bool bACTEnabled = m_pcEncCfg->getRGBFormatFlag()? (!singleColorSpaceId? true: false): (singleColorSpaceId? true: false);
    pcCU->setColourTransformSubParts(bACTEnabled, absPartIdx, fullDepth);
    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      ComponentID compID = ComponentID(ch);
      pcCU->setTransformSkipPartRange(singleComponentTransformMode[singleColorSpaceId][compID],                      compID, absPartIdx, partIndxNumPerTU );
      pcCU->setCbfPartRange          ((singleComponentCbf[singleColorSpaceId][compID] << trDepth),                 compID, absPartIdx, partIndxNumPerTU );
      pcCU->setCrossComponentPredictionAlphaPartRange(cSingleComponentPredictionAlpha[singleColorSpaceId][compID],     compID, absPartIdx, partIndxNumPerTU );
    }

    xLoadIntraResultQT(COMPONENT_Y,  rTu, true );
    xLoadIntraResultQT(COMPONENT_Cb, rTu, true );
    xLoadIntraResultQT(COMPONENT_Cr, rTu, true );
    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_CHROMA_INTRA ] );

    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      ComponentID         compID           = ComponentID(ch);
      const TComRectangle &rect            = rTu.getRect(compID);
      const UInt          width            = rect.width;
      const UInt          height           = rect.height;
      const UInt          ZOrder           = pcCU->getZorderIdxInCtu() + absPartIdx;
      Pel*                pcRecQt          = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
      Pel*                piRecIPred       = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
      const UInt          recQtStride      = m_pcQTTempTComYuv[ QTLayer ].getStride(compID);
      const UInt          recIPredStride   = pcCU->getPic()->getPicYuvRec()->getStride (compID);
      for( UInt y = 0; y < height; y++ )
      {
        for( UInt x = 0; x < width; x++ )
        {
          piRecIPred[ x ] = pcRecQt[ x ];
        }
        pcRecQt    += recQtStride;
        piRecIPred += recIPredStride;
      }
    }
  }

  if( m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled != MAX_DOUBLE && m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled != MAX_DOUBLE )
  {
    if( m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled < m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled )
    {
      m_sACTRDCostTU[relTrDepth].uiIsCSCEnabled = 0;
    }
    else
    {
      m_sACTRDCostTU[relTrDepth].uiIsCSCEnabled = 1;
    }
  }
  else if( m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled == MAX_DOUBLE && m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled == MAX_DOUBLE )
  {
    m_sACTRDCostTU[relTrDepth].uiIsCSCEnabled = 2;
  }
  else if( m_sACTRDCostTU[relTrDepth].tmpRDCostCSCEnabled != MAX_DOUBLE && m_sACTRDCostTU[relTrDepth].tmpRDCostCSCDisabled == MAX_DOUBLE )
  {
    m_sACTRDCostTU[relTrDepth].uiIsCSCEnabled = 1;
  }
  else
  {
    assert(0);
  }

  if( bCheckSplit )
  {
    if( bCheckFull )
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    }
    else
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    }

    Double     dSplitCost                       = 0.0;
    Distortion splitDist[MAX_NUM_CHANNEL_TYPE]  = {0, 0};
    UInt       splitBits                        = 0;
    UInt       splitCbf[MAX_NUM_COMPONENT]      = {0, 0, 0};

    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xRecurIntraCodingQTTUCSC( pcOrgYuv, pcPredYuv, pcResiYuv, splitDist[CHANNEL_TYPE_LUMA], splitDist[CHANNEL_TYPE_CHROMA], dSplitCost, tuRecurseChild, bTestMaxTUSize, eACTRDTestType DEBUG_STRING_PASS_INTO(sDebug) );

      for(UInt ch = 0; ch < numberValidComponents; ch++)
      {
        splitCbf[ch] |= pcCU->getCbf( tuRecurseChild.GetAbsPartIdxTU(), ComponentID(ch), tuRecurseChild.GetTransformDepthRel() );
      }
    } while ( tuRecurseChild.nextSection(rTu) );

    UInt partsDiv = rTu.GetAbsPartIdxNumParts();
    for(UInt ch = 0; ch < numberValidComponents; ch++)
    {
      if (splitCbf[ch])
      {
        const UInt        flag   = 1 << trDepth;
        const ComponentID compID = ComponentID(ch);
        UChar             *pBase = pcCU->getCbf( compID );
        for( UInt offs = 0; offs < partsDiv; offs++ )
        {
          pBase[ absPartIdx + offs ] |= flag;
        }
      }
    }

    m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    m_pcEntropyCoder->resetBits();

    splitBits += xGetIntraBitsQT( rTu, true, true, false );
    dSplitCost  = m_pcRdCost->calcRdCost( splitBits, splitDist[CHANNEL_TYPE_LUMA] + splitDist[CHANNEL_TYPE_CHROMA] );

    if( dSplitCost < dSingleCost )
    {
      PUDistY += splitDist[CHANNEL_TYPE_LUMA];
      PUDistC += splitDist[CHANNEL_TYPE_CHROMA];
      dPUCost += dSplitCost;
      return;
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );

    pcCU->setTrIdxSubParts( trDepth, absPartIdx, fullDepth );
    Bool bACTEnabled = m_pcEncCfg->getRGBFormatFlag()? (!singleColorSpaceId? true: false): (singleColorSpaceId? true: false);
    pcCU->setColourTransformSubParts(bACTEnabled, absPartIdx, fullDepth);

    for(UInt ch = 0; ch < numberValidComponents; ch++)
    {
      const ComponentID   compID            = ComponentID(ch);
      const TComRectangle &tuRect           = rTu.getRect(compID);
      const UInt          subTUAbsPartIdx   = rTu.GetAbsPartIdxTU(compID);
      const UInt          partIdxesPerSubTU = rTu.GetAbsPartIdxNumParts(compID);
      assert(subTUAbsPartIdx == absPartIdx);

      pcCU->setTransformSkipPartRange(singleComponentTransformMode[singleColorSpaceId][compID], compID, subTUAbsPartIdx, partIdxesPerSubTU );
      pcCU->setCbfPartRange( (singleComponentCbf[singleColorSpaceId][compID] << trDepth), compID, subTUAbsPartIdx, partIdxesPerSubTU );
      pcCU->setCrossComponentPredictionAlphaPartRange(cSingleComponentPredictionAlpha[singleColorSpaceId][compID], compID, subTUAbsPartIdx, partIdxesPerSubTU );

      //--- set reconstruction for next intra TU
      const UInt  ZOrder    = pcCU->getZorderIdxInCtu() + absPartIdx;
      const UInt  width     = tuRect.width;
      const UInt  height    = tuRect.height;
      Pel*        piSrc     = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
      UInt        srcStride = m_pcQTTempTComYuv[ QTLayer ].getStride  ( compID );
      Pel*        piDes     = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
      UInt        desStride = pcCU->getPic()->getPicYuvRec()->getStride  ( compID );

      for( UInt y = 0; y < height; y++, piSrc += srcStride, piDes += desStride )
      {
        for( UInt x = 0; x < width; x++ )
        {
          piDes[ x ] = piSrc[ x ];
        }
      }
    }
  }
  PUDistY += singleDist[CHANNEL_TYPE_LUMA];
  PUDistC += singleDist[CHANNEL_TYPE_CHROMA];
  dPUCost   += dSingleCost;
}

Void
TEncSearch::xRecurIntraCodingQTCSC( TComYuv* pcOrgYuv, TComYuv* pcPredYuv, TComYuv* pcResiYuv, Distortion& PUDistY, Distortion& PUDistC, Double& dPUCost, TComTU& rTu, Bool bTestMaxTUSize DEBUG_STRING_FN_DECLARE(sDebug) )
{
  TComDataCU          *pcCU                 = rTu.getCU();
  const UInt          trDepth               = rTu.GetTransformDepthRel();
  const UInt          fullDepth             = rTu.GetTransformDepthTotal();
  const UInt          absPartIdx            = rTu.GetAbsPartIdxTU();
  const UInt          partIndxNumPerTU      = rTu.GetAbsPartIdxNumParts(COMPONENT_Y);
  const ChromaFormat  chFmt                 = rTu.GetChromaFormat();
  const UInt          log2TrSize            = rTu.GetLog2LumaTrSize();
  const UInt          QTLayer               = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - log2TrSize;
  const UInt          numberValidComponents = getNumberValidComponents(chFmt);
  Bool                bCheckFull            = ( log2TrSize  <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() );
  Bool                bCheckSplit           = ( bTestMaxTUSize && bCheckFull )? false: ( log2TrSize  >  pcCU->getQuadtreeTULog2MinSizeInCU(absPartIdx) );
  const Bool          extendedPrecision     = rTu.getCU()->getSlice()->getSPS()->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();

  UInt depth = pcCU->getDepth(absPartIdx);

  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(absPartIdx) && bCheckFull)
  {
    bCheckSplit = false;
  }
  else if ( m_pcEncCfg->getNoTUSplitIntraACTEnabled() && depth <= 1 && bCheckFull )
  {
    bCheckSplit = false;
  }

  assert( bCheckFull || bCheckSplit );
  assert( chFmt == CHROMA_444);

  Double              dSingleCost                                 = MAX_DOUBLE;
  Distortion          singleDist[MAX_NUM_CHANNEL_TYPE]            = {0, 0};
  UInt                singleBits                                  = 0;

  Double              dSingleComponentCost[MAX_NUM_COMPONENT]            = {MAX_DOUBLE, MAX_DOUBLE, MAX_DOUBLE};
  UInt                singleComponentCbf[MAX_NUM_COMPONENT]              = {0, 0, 0};
  UInt                singleComponentTransformMode[MAX_NUM_COMPONENT]    = {0, 0, 0};
  SChar               cSingleComponentPredictionAlpha[MAX_NUM_COMPONENT] = {0, 0, 0};

  if( bCheckFull )
  {
    m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );

    pcCU->setTrIdxSubParts( trDepth, absPartIdx, fullDepth );

    //intra prediction
    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      const ComponentID   compID    = ComponentID(ch);
      const ChannelType   chType    = toChannelType(compID);
      const TComRectangle &rect     = rTu.getRect(compID);
      const UInt          width     = rect.width;
      const UInt          height    = rect.height;

      const Bool bIsLuma            = isLuma(compID);
      const UInt chPredMode         = pcCU->getIntraDir( chType, absPartIdx );
      const UInt chFinalMode        = (chPredMode == DM_CHROMA_IDX && !bIsLuma) ? pcCU->getIntraDir(CHANNEL_TYPE_LUMA, absPartIdx) : chPredMode;

      const Bool bUseFilteredPredictions = TComPrediction::filteringIntraReferenceSamples(compID, chFinalMode, width, height, chFmt, pcCU->getSlice()->getSPS()->getSpsRangeExtension().getIntraSmoothingDisabledFlag());
      Pel*       piOrg                   = pcOrgYuv ->getAddr( compID, absPartIdx );
      Pel*       piPred                  = pcPredYuv->getAddr( compID, absPartIdx );
      Pel*       piResi                  = pcResiYuv->getAddr( compID, absPartIdx );
      const UInt stride                  = pcOrgYuv ->getStride (compID);

      DEBUG_STRING_NEW(sTemp)
      initIntraPatternChType( rTu, compID, bUseFilteredPredictions DEBUG_STRING_PASS_INTO(sTemp) );
      predIntraAng( compID, chFinalMode, piOrg, stride, piPred, stride, rTu, bUseFilteredPredictions );

      Pel* pcOrg  = piOrg;
      Pel* pcPred = piPred;
      Pel* pcResi = piResi;

      for( UInt y = 0; y < height; y++ )
      {
        for ( UInt x = 0; x < width; x++ )
        {
          pcResi[x] = pcOrg[x] - pcPred[x];
        }

        pcOrg  += stride;
        pcResi += stride;
        pcPred += stride;
      }
    }

    pcResiYuv->convert(extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, true, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0));
    SChar preCalcAlpha[MAX_NUM_COMPONENT] = {0, 0, 0};

    if( pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag() && !m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate() )
    {
      for( UInt ch = 0; ch < numberValidComponents; ch++ )
      {
        const ComponentID    compID                        = ComponentID(ch);
        const TComRectangle &tuCompRect                    = rTu.getRect(compID);
        const Pel  *const    lumaResidualForEstimate       = pcResiYuv->getAddr( COMPONENT_Y, absPartIdx );
        const UInt           lumaResidualStrideForEstimate = pcResiYuv->getStride ( COMPONENT_Y );

        const UInt           chPredMode                  = pcCU->getIntraDir( toChannelType(compID), absPartIdx );
        const Bool bDecorrelationAvailable                 = isChroma(compID) && (chPredMode == DM_CHROMA_IDX);
        if( bDecorrelationAvailable )
        {
          preCalcAlpha[compID] = xCalcCrossComponentPredictionAlpha(rTu,
                                                                    compID,
                                                                    lumaResidualForEstimate,
                                                                    pcResiYuv->getAddr( compID, absPartIdx ),
                                                                    tuCompRect.width,
                                                                    tuCompRect.height,
                                                                    lumaResidualStrideForEstimate,
                                                                    pcResiYuv->getStride( compID )
                                                                  );
        }
      }
    }

    Pel reconResiLuma   [MAX_CU_SIZE * MAX_CU_SIZE];
    Pel reconResiLumaTmp[MAX_CU_SIZE * MAX_CU_SIZE];

    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      Double     dSingleCostTmp = MAX_DOUBLE;
      UInt       singleBitsTmp  = 0;
      Distortion singleDistTmp  = 0;
      UInt       singleCbfTmp   = 0;

      const ComponentID    compID     = ComponentID(ch);
      const TComRectangle &tuCompRect = rTu.getRect(compID);
      assert(pcCU->getColourTransform(absPartIdx));
      QpParam cQP(*pcCU, compID, absPartIdx);
      if(!pcCU->isLosslessCoded(0))
      {
        Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
        m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( deltaQP );
        m_pcRdCost->adjustLambdaForColourTrans( deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
      }

      Bool bCheckTransformSkip = pcCU->getSlice()->getPPS()->getUseTransformSkip() &&
        TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(compID), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize()) &&
        (!pcCU->isLosslessCoded(0));
      if( m_pcEncCfg->getUseTransformSkipFast() )
      {
        bCheckTransformSkip &= (pcCU->getPartitionSize(absPartIdx) == SIZE_NxN);
        if ( isChroma( compID ) )
        {
          bCheckTransformSkip &= (pcCU->getTransformSkip( absPartIdx, COMPONENT_Y ) != 0);
        }
      }

      Bool bCheckCrossComponentPrediction =    isChroma(compID)
                                            && pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag()
                                            && (pcCU->getIntraDir( toChannelType(compID), absPartIdx ) == DM_CHROMA_IDX)
                                            && (pcCU->getCbf(absPartIdx, COMPONENT_Y, trDepth) != 0);

      SChar cCalcAlpha = 0;
      if ( bCheckCrossComponentPrediction && m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate() )
      {
        cCalcAlpha = xCalcCrossComponentPredictionAlpha(rTu,
                                                        compID,
                                                        reconResiLuma + tuCompRect.y0 * MAX_CU_SIZE + tuCompRect.x0,
                                                        pcResiYuv->getAddr( compID, absPartIdx),
                                                        tuCompRect.width,
                                                        tuCompRect.height,
                                                        MAX_CU_SIZE,
                                                        pcResiYuv->getStride( compID )
                                                      );
      }
      else if ( bCheckCrossComponentPrediction )
      {
        cCalcAlpha = preCalcAlpha[compID];
      }
      bCheckCrossComponentPrediction &= (cCalcAlpha != 0);

      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );

      const Int transformSkipModesToTest    = bCheckTransformSkip               ? 2 : 1;
      const Int crossCPredictionModesToTest = bCheckCrossComponentPrediction    ? 2 : 1;

      for ( Int transformSkipModeId = 0; transformSkipModeId < transformSkipModesToTest; transformSkipModeId++ )
      {
        for ( Int crossCPredictionModeId = 0; crossCPredictionModeId < crossCPredictionModesToTest; crossCPredictionModeId++ )
        {
          DEBUG_STRING_NEW(sModeString)
          m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[fullDepth][CI_QT_TRAFO_TEST] );
          m_pcEntropyCoder->resetBits();

          pcCU->setTransformSkipPartRange( transformSkipModeId, compID, absPartIdx, partIndxNumPerTU );
          pcCU->setCrossComponentPredictionAlphaPartRange( (crossCPredictionModeId != 0) ? cCalcAlpha : 0, compID, absPartIdx, partIndxNumPerTU );

          xIntraCodingTUBlockCSC( pcResiYuv, (compID == COMPONENT_Y) ? reconResiLumaTmp : reconResiLuma, (crossCPredictionModeId != 0), compID, rTu, cQP DEBUG_STRING_PASS_INTO(sModeString) );
          singleCbfTmp = pcCU->getCbf( absPartIdx, compID, trDepth );

          if ( (singleCbfTmp == 0) && (transformSkipModeId != 0) )
          {
            dSingleCostTmp = MAX_DOUBLE;
          }
          else
          {
            if ( compID == COMPONENT_Y )
            {
              singleBitsTmp = xGetIntraBitsQT( rTu, true, false, false );
            }
            else
            {
              singleBitsTmp = xGetIntraBitsQTChroma( rTu, compID, false );
            }

            //residual distortion in YCgCo domain
            singleDistTmp = m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType( compID )],
                                                     m_pcQTTempTComYuv[QTLayer].getAddr( compID, absPartIdx ),
                                                     m_pcQTTempTComYuv[QTLayer].getStride( compID ),
                                                     pcResiYuv->getAddr( compID, absPartIdx ),
                                                     pcResiYuv->getStride( compID ),
                                                     rTu.getRect( compID ).width,
                                                     rTu.getRect( compID ).height,
                                                     compID );
            dSingleCostTmp = m_pcRdCost->calcRdCost( singleBitsTmp, singleDistTmp );
          }

          if ( dSingleCostTmp < dSingleComponentCost[compID] )
          {
            dSingleComponentCost[compID]             = dSingleCostTmp;
            singleComponentCbf[compID]             = singleCbfTmp;
            cSingleComponentPredictionAlpha[compID]  = (crossCPredictionModeId != 0) ? pcCU->getCrossComponentPredictionAlpha( absPartIdx, compID ) : 0;
            singleComponentTransformMode[compID]   = transformSkipModeId;

            xStoreIntraResultQT( compID, rTu );

            m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[fullDepth][CI_TEMP_BEST] );

            if ( compID == COMPONENT_Y )
            {
              const Int xOffset = rTu.getRect( COMPONENT_Y ).x0;
              const Int yOffset = rTu.getRect( COMPONENT_Y ).y0;
              xStoreCrossComponentPredictionResult( reconResiLuma, reconResiLumaTmp, rTu, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE );
            }
          }
        }
      }

      pcCU->setTransformSkipPartRange(singleComponentTransformMode[compID],                      compID, absPartIdx, partIndxNumPerTU );
      pcCU->setCbfPartRange          ((singleComponentCbf[compID] << trDepth),                 compID, absPartIdx, partIndxNumPerTU );
      pcCU->setCrossComponentPredictionAlphaPartRange(cSingleComponentPredictionAlpha[compID],     compID, absPartIdx, partIndxNumPerTU );

      xLoadIntraResultQT(compID, rTu );

      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_TEMP_BEST ] );
      if(!pcCU->isLosslessCoded(0))
      {
        Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
        m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( - deltaQP );
        m_pcRdCost->adjustLambdaForColourTrans            ( - deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
      }
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    m_pcEntropyCoder->resetBits();

    singleBits = xGetIntraBitsQT( rTu, true, true, false );

    m_pcQTTempTComYuv[ QTLayer ].convert(extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, false, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(0));

    for( UInt ch = 0; ch < numberValidComponents; ch++ )
    {
      ComponentID         compID           = ComponentID(ch);
      const TComRectangle &rect            = rTu.getRect(compID);
      const UInt          width            = rect.width;
      const UInt          height           = rect.height;
      const UInt          ZOrder           = pcCU->getZorderIdxInCtu() + absPartIdx;
      Pel*                pcPred           = pcPredYuv->getAddr( compID, absPartIdx );
      Pel*                pcResi           = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
      Pel*                pcRecQt          = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
      Pel*                piRecIPred       = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
      const UInt          stride           = pcPredYuv->getStride (compID);
      const UInt          recQtStride      = m_pcQTTempTComYuv[ QTLayer ].getStride(compID);
      const UInt          recIPredStride   = pcCU->getPic()->getPicYuvRec()->getStride (compID);
      const UInt          clipbd           = pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(compID)];

      for( UInt y = 0; y < height; y++ )
      {
        for( UInt x = 0; x < width; x++ )
        {
          pcRecQt[ x ] = Pel( ClipBD<Int>( Int(pcPred[x]) + Int(pcResi[x]), clipbd ) );
          piRecIPred[ x ] = pcRecQt[ x ];
        }

        pcPred     += stride;
        pcResi     += recQtStride;
        pcRecQt    += recQtStride;
        piRecIPred += recIPredStride;
      }
    }

    singleDist[CHANNEL_TYPE_LUMA] = m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(COMPONENT_Y)],
                                                             m_pcQTTempTComYuv[QTLayer].getAddr(COMPONENT_Y, absPartIdx),
                                                             m_pcQTTempTComYuv[QTLayer].getStride(COMPONENT_Y),
                                                             pcOrgYuv->getAddr(COMPONENT_Y, absPartIdx),
                                                             pcOrgYuv->getStride(COMPONENT_Y),
                                                             rTu.getRect(COMPONENT_Y).width,
                                                             rTu.getRect(COMPONENT_Y).height,
                                                             COMPONENT_Y);
    singleDist[CHANNEL_TYPE_CHROMA] = 0;
    for(UInt ch = COMPONENT_Cb; ch < numberValidComponents; ch++)
    {
      ComponentID compID = ComponentID(ch);
      singleDist[CHANNEL_TYPE_CHROMA] += m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepths().recon[toChannelType(compID)],
                                                                  m_pcQTTempTComYuv[QTLayer].getAddr(compID, absPartIdx),
                                                                  m_pcQTTempTComYuv[QTLayer].getStride(compID),
                                                                  pcOrgYuv->getAddr(compID, absPartIdx),
                                                                  pcOrgYuv->getStride(compID),
                                                                  rTu.getRect(compID).width,
                                                                  rTu.getRect(compID).height,
                                                                  compID);
    }

    dSingleCost = m_pcRdCost->calcRdCost( singleBits, singleDist[CHANNEL_TYPE_LUMA] + singleDist[CHANNEL_TYPE_CHROMA] );
  }

  if( bCheckSplit )
  {
    if( bCheckFull )
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    }
    else
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[fullDepth][CI_QT_TRAFO_ROOT] );
    }

    Double     dSplitCost                       = 0.0;
    Distortion splitDist[MAX_NUM_CHANNEL_TYPE]  = {0, 0};
    UInt       splitBits                        = 0;
    UInt       splitCbf[MAX_NUM_COMPONENT]      = {0, 0, 0};

    TComTURecurse tuRecurseChild(rTu, false);
    do
    {
      xRecurIntraCodingQTCSC( pcOrgYuv, pcPredYuv, pcResiYuv, splitDist[CHANNEL_TYPE_LUMA], splitDist[CHANNEL_TYPE_CHROMA], dSplitCost, tuRecurseChild, bTestMaxTUSize DEBUG_STRING_PASS_INTO(sDebug) );

      for(UInt ch = 0; ch < numberValidComponents; ch++)
      {
        splitCbf[ch] |= pcCU->getCbf( tuRecurseChild.GetAbsPartIdxTU(), ComponentID(ch), tuRecurseChild.GetTransformDepthRel() );
      }
    } while ( tuRecurseChild.nextSection(rTu) );

    UInt uiPartsDiv = rTu.GetAbsPartIdxNumParts();
    for(UInt ch = 0; ch < numberValidComponents; ch++)
    {
      if (splitCbf[ch])
      {
        const UInt        flag   = 1 << trDepth;
        const ComponentID compID = ComponentID(ch);
        UChar             *pBase = pcCU->getCbf( compID );
        for ( UInt uiOffs = 0; uiOffs < uiPartsDiv; uiOffs++ )
        {
          pBase[absPartIdx + uiOffs] |= flag;
        }
      }
    }

    m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_ROOT ] );
    m_pcEntropyCoder->resetBits();

    splitBits += xGetIntraBitsQT( rTu, true, true, false );
    dSplitCost  = m_pcRdCost->calcRdCost( splitBits, splitDist[CHANNEL_TYPE_LUMA] + splitDist[CHANNEL_TYPE_CHROMA] );

    if( dSplitCost < dSingleCost )
    {
      PUDistY += splitDist[CHANNEL_TYPE_LUMA];
      PUDistC += splitDist[CHANNEL_TYPE_CHROMA];
      dPUCost   += dSplitCost;
      return;
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ fullDepth ][ CI_QT_TRAFO_TEST ] );

    pcCU->setTrIdxSubParts( trDepth, absPartIdx, fullDepth );

    for(UInt ch = 0; ch < numberValidComponents; ch++)
    {
      const ComponentID   compID            = ComponentID(ch);
      const TComRectangle &tuRect           = rTu.getRect(compID);
      const UInt          subTUAbsPartIdx   = rTu.GetAbsPartIdxTU(compID);
      const UInt          partIdxesPerSubTU = rTu.GetAbsPartIdxNumParts(compID);
      assert(subTUAbsPartIdx == absPartIdx);

      pcCU->setTransformSkipPartRange(singleComponentTransformMode[compID], compID, subTUAbsPartIdx, partIdxesPerSubTU );
      pcCU->setCbfPartRange( (singleComponentCbf[compID] << trDepth), compID, subTUAbsPartIdx, partIdxesPerSubTU );
      pcCU->setCrossComponentPredictionAlphaPartRange(cSingleComponentPredictionAlpha[compID], compID, subTUAbsPartIdx, partIdxesPerSubTU );

      //--- set reconstruction for next intra TU
      const UInt  ZOrder    = pcCU->getZorderIdxInCtu() + absPartIdx;
      const UInt  width     = tuRect.width;
      const UInt  height    = tuRect.height;
      Pel*        piSrc     = m_pcQTTempTComYuv[ QTLayer ].getAddr( compID, absPartIdx );
      UInt        srcStride = m_pcQTTempTComYuv[ QTLayer ].getStride  ( compID );
      Pel*        piDes     = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
      UInt        desStride = pcCU->getPic()->getPicYuvRec()->getStride  ( compID );

      for( UInt y = 0; y < height; y++, piSrc += srcStride, piDes += desStride )
      {
        for ( UInt x = 0; x < width; x++ )
        {
          piDes[x] = piSrc[x];
        }
      }
    }
  }
  PUDistY += singleDist[CHANNEL_TYPE_LUMA];
  PUDistC += singleDist[CHANNEL_TYPE_CHROMA];
  dPUCost   += dSingleCost;
}

Void
TEncSearch::estIntraPredQTCT( TComDataCU*    pcCU,
                              TComYuv*       pcOrgYuv,
                              TComYuv*       pcPredYuv,
                              TComYuv*       pcResiYuv,
                              TComYuv*       pcRecoYuv,
                              ACTRDTestTypes eACTRDTestType,
                              Bool           bReuseIntraMode
                              DEBUG_STRING_FN_DECLARE(sDebug)
                             )
{
  const UInt         depth                 = pcCU->getDepth(0);
  const UInt         initTrDepth           = pcCU->getPartitionSize(0) == SIZE_2Nx2N ? 0 : 1;
  const UInt         numPU                 = 1<<(2*initTrDepth);
  const UInt         QNumParts             = pcCU->getTotalNumPart() >> 2;
  const ChromaFormat chFmt                 = pcCU->getPic()->getChromaFormat();
  const UInt         widthBit              = pcCU->getIntraSizeIdx(0);
  const UInt         numberValidComponents = getNumberValidComponents(chFmt);
  Distortion         overallDistY          = 0;
  Distortion         overallDistC          = 0;
  const TComSPS     &sps                   = *(pcCU->getSlice()->getSPS());

#if FULL_NBIT
  const Double sqrtLambdaForFirstPass= (m_pcEncCfg->getCostMode()==COST_MIXED_LOSSLESS_LOSSY_CODING && pcCU->getCUTransquantBypass(0)) ?
    sqrt(0.57 * pow(2.0, ((LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME - 12) / 3.0)))
    : m_pcRdCost->getSqrtLambda();
#else
  const Double sqrtLambdaForFirstPass= (m_pcEncCfg->getCostMode()==COST_MIXED_LOSSLESS_LOSSY_CODING && pcCU->getCUTransquantBypass(0)) ?
    sqrt(0.57 * pow(2.0, ((LOSSLESS_AND_MIXED_LOSSLESS_RD_COST_TEST_QP_PRIME - 12 - 6 * (pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) - 8)) / 3.0)))
    : m_pcRdCost->getSqrtLambda();
#endif

  if ( pcCU->getSlice()->getPPS()->getUseDQP() == true )
  {
    pcCU->setQPSubParts( pcCU->getQP( 0 ), 0, depth );
  }
  else
  {
    pcCU->setQPSubParts( pcCU->getSlice()->getSliceQp(), 0, depth );
  }

  //===== loop over partitions =====
  TComTURecurse tuRecurseCU(pcCU, 0);
  TComTURecurse tuRecurseWithPU(tuRecurseCU, false, (initTrDepth==0)?TComTU::DONT_SPLIT : TComTU::QUAD_SPLIT);

  do
  {
    //luma intra mode selection
    UInt                uiRdModeList[35];
    Double              CandCostList[35];
    Int                 numModesAvailable = 35;
    Int                 numModesForFullRD = m_pcEncCfg->getFastUDIUseMPMEnabled() ? g_aucIntraModeNumFast_UseMPM[ widthBit ] : g_aucIntraModeNumFast_NotUseMPM[ widthBit ];
    const TComRectangle &puRect           = tuRecurseWithPU.getRect(COMPONENT_Y);
    const UInt          partOffset      = tuRecurseWithPU.GetAbsPartIdxTU();
    const UInt          uiLog2TrSize      = tuRecurseWithPU.GetLog2LumaTrSize();
    UInt                CandNum;

    Pel*                piOrg             = pcOrgYuv ->getAddr( COMPONENT_Y, partOffset );
    Pel*                piPred            = pcPredYuv->getAddr( COMPONENT_Y, partOffset );
    UInt                stride            = pcPredYuv->getStride( COMPONENT_Y );

    DEBUG_STRING_NEW(sTemp2)

    for ( Int i=0; i < numModesForFullRD; i++ )
    {
      CandCostList[i] = MAX_DOUBLE;
    }
    CandNum = 0;

    initIntraPatternChType( tuRecurseWithPU, COMPONENT_Y, true DEBUG_STRING_PASS_INTO(sTemp2) );

    if(bReuseIntraMode)
    {
      uiRdModeList [0] =tuRecurseWithPU.getCU()->getIntraDir( CHANNEL_TYPE_LUMA, tuRecurseWithPU.GetAbsPartIdxTU() );
      numModesForFullRD = 1;
    }
    else
    {
      DistParam distParam;
      const Bool bUseHadamard=pcCU->getCUTransquantBypass(0) == 0;
      m_pcRdCost->setDistParam(distParam, sps.getBitDepth(CHANNEL_TYPE_LUMA), piOrg, stride, piPred, stride, puRect.width, puRect.height, bUseHadamard);
      distParam.bApplyWeight = false;
      for( Int modeIdx = 0; modeIdx < numModesAvailable; modeIdx++ )
      {
        UInt       mode = modeIdx;
        Distortion sad  = 0;

        const Bool bUseFilter = TComPrediction::filteringIntraReferenceSamples( COMPONENT_Y, mode, puRect.width, puRect.height, chFmt, pcCU->getSlice()->getSPS()->getSpsRangeExtension().getIntraSmoothingDisabledFlag() );
        predIntraAng( COMPONENT_Y, mode, piOrg, stride, piPred, stride, tuRecurseWithPU, bUseFilter, TComPrediction::UseDPCMForFirstPassIntraEstimation(tuRecurseWithPU, mode) );

        //hadamard transform
        sad += distParam.DistFunc(&distParam);

        UInt iModeBits = 0;
        iModeBits     += xModeBitsIntra( pcCU, mode, partOffset, depth, CHANNEL_TYPE_LUMA );
        Double cost    = (Double)sad + (Double)iModeBits * sqrtLambdaForFirstPass;

        CandNum += xUpdateCandList( mode, cost, numModesForFullRD, uiRdModeList, CandCostList );
      }

      Int preds[NUM_MOST_PROBABLE_MODES] = {-1, -1, -1};
      Int iMode                            = -1;
      pcCU->getIntraDirPredictor( partOffset, preds, COMPONENT_Y, &iMode );
      const Int numCand = ( iMode >= 0 ) ? iMode : Int(NUM_MOST_PROBABLE_MODES);

      for( Int j=0; j < numCand; j++)
      {
        Bool mostProbableModeIncluded = false;
        Int  mostProbableMode         = preds[j];

        for ( Int i=0; i < numModesForFullRD; i++ )
        {
          mostProbableModeIncluded |= (mostProbableMode == uiRdModeList[i]);
        }

        if ( !mostProbableModeIncluded )
        {
          uiRdModeList[numModesForFullRD++] = mostProbableMode;
        }
      }
    }
    UInt       bestPUMode[MAX_NUM_CHANNEL_TYPE]  = {0, 0};
    Distortion bestPUDist[MAX_NUM_CHANNEL_TYPE]  = {0, 0};
    Double     dBestPUCost                       = MAX_DOUBLE;

    //select luma intra mode
    for( UInt lumaModeIdx = 0; lumaModeIdx < numModesForFullRD; lumaModeIdx++ )  //candidate luma intra mode
    {
      pcCU->setIntraDirSubParts( CHANNEL_TYPE_LUMA,   uiRdModeList[lumaModeIdx], partOffset, depth + initTrDepth );
      pcCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, DM_CHROMA_IDX,             partOffset, depth + initTrDepth );  //use DM_CHROMA_IDX for chroma intra mode
      if( pcCU->getPartitionSize(0) == SIZE_NxN )
      {
        pcCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, DM_CHROMA_IDX, 0, depth );
      }

      DEBUG_STRING_NEW(sMode)
      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[depth][CI_CURR_BEST] );

      Distortion PUDistY = 0;
      Distortion PUDistC = 0;
      Double     dPUCost = 0;

      if(eACTRDTestType == ACT_TWO_CLR)
      {
        xRecurIntraCodingQTTUCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, true, eACTRDTestType DEBUG_STRING_PASS_INTO(sMode)  );
      }
      else if(eACTRDTestType == ACT_TRAN_CLR)
      {
        pcCU->setColourTransformSubParts(true, 0, depth);
        xRecurIntraCodingQTCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, true DEBUG_STRING_PASS_INTO(sMode)  );
      }
      else if(eACTRDTestType == ACT_ORG_CLR)
      {
        xRecurIntraCodingQTTUCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, true, eACTRDTestType DEBUG_STRING_PASS_INTO(sMode)  );
      }
      else
      {
        assert(0);
      }

      if( dPUCost < dBestPUCost)
      {
        dBestPUCost                     = dPUCost;
        bestPUDist[CHANNEL_TYPE_LUMA]   = PUDistY;
        bestPUDist[CHANNEL_TYPE_CHROMA] = PUDistC;
        bestPUMode[CHANNEL_TYPE_LUMA]   = uiRdModeList[lumaModeIdx];
        bestPUMode[CHANNEL_TYPE_CHROMA] = DM_CHROMA_IDX;
        xSetIntraResultLumaQT( pcRecoYuv, tuRecurseWithPU );
        xSetIntraResultChromaQT( pcRecoYuv, tuRecurseWithPU );

        UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();
        ::memcpy( m_puhQTTempTrIdx,  pcCU->getTransformIdx() + partOffset, uiQPartNum * sizeof( UChar ) );
        ::memcpy( m_puhQTTempACTFlag, pcCU->getColourTransform() + partOffset, uiQPartNum * sizeof(Bool) );
        for (UInt component = 0; component < numberValidComponents; component++)
        {
          const ComponentID compID = ComponentID(component);
          ::memcpy( m_puhQTTempCbf[compID],                             pcCU->getCbf( compID  )                           + partOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_puhQTTempTransformSkipFlag[compID],               pcCU->getTransformSkip(compID)                    + partOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_phQTTempCrossComponentPredictionAlpha[compID],    pcCU->getCrossComponentPredictionAlpha(compID)    + partOffset, uiQPartNum * sizeof( SChar ) );
        }
      }
    }

    if ((!m_pcEncCfg->getTransquantBypassInferTUSplit() || !pcCU->isLosslessCoded(0)) && (uiLog2TrSize > pcCU->getQuadtreeTULog2MinSizeInCU(partOffset)) && ((!m_pcEncCfg->getNoTUSplitIntraACTEnabled()) || (m_pcEncCfg->getNoTUSplitIntraACTEnabled() && depth > 1)))
    {
      Distortion PUDistY = 0;
      Distortion PUDistC = 0;
      Double     dPUCost   = 0;

      pcCU->setIntraDirSubParts( CHANNEL_TYPE_LUMA,   bestPUMode[CHANNEL_TYPE_LUMA],   partOffset, depth + initTrDepth );
      pcCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, bestPUMode[CHANNEL_TYPE_CHROMA], partOffset, depth + initTrDepth );

      DEBUG_STRING_NEW(sMode)
      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[depth][CI_CURR_BEST] );
      if(eACTRDTestType == ACT_TWO_CLR)
      {
        xRecurIntraCodingQTTUCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, false, eACTRDTestType DEBUG_STRING_PASS_INTO(sMode) );
      }
      else if(eACTRDTestType == ACT_TRAN_CLR)
      {
        pcCU->setColourTransformSubParts(true, 0, depth);
        xRecurIntraCodingQTCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, false DEBUG_STRING_PASS_INTO(sMode) );
      }
      else if(eACTRDTestType == ACT_ORG_CLR)
      {
        xRecurIntraCodingQTTUCSC( pcOrgYuv, pcPredYuv, pcResiYuv, PUDistY, PUDistC, dPUCost, tuRecurseWithPU, false, eACTRDTestType DEBUG_STRING_PASS_INTO(sMode) );
      }
      else
      {
        assert(0);
      }

      if( dPUCost < dBestPUCost)
      {
        dBestPUCost                     = dPUCost;
        bestPUDist[CHANNEL_TYPE_LUMA]   = PUDistY;
        bestPUDist[CHANNEL_TYPE_CHROMA] = PUDistC;
        xSetIntraResultLumaQT( pcRecoYuv, tuRecurseWithPU );
        xSetIntraResultChromaQT( pcRecoYuv, tuRecurseWithPU );

        UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();
        ::memcpy( m_puhQTTempTrIdx,  pcCU->getTransformIdx() + partOffset, uiQPartNum * sizeof( UChar ) );
        ::memcpy( m_puhQTTempACTFlag, pcCU->getColourTransform() + partOffset, uiQPartNum * sizeof(Bool) );
        for (UInt component = 0; component < numberValidComponents; component++)
        {
          const ComponentID compID = ComponentID(component);
          ::memcpy( m_puhQTTempCbf[compID],                             pcCU->getCbf( compID  )                           + partOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_puhQTTempTransformSkipFlag[compID],               pcCU->getTransformSkip(compID)                    + partOffset, uiQPartNum * sizeof( UChar ) );
          ::memcpy( m_phQTTempCrossComponentPredictionAlpha[compID],    pcCU->getCrossComponentPredictionAlpha(compID)    + partOffset, uiQPartNum * sizeof( SChar ) );
        }
      }
    }

    overallDistY += bestPUDist[CHANNEL_TYPE_LUMA];
    overallDistC += bestPUDist[CHANNEL_TYPE_CHROMA];

    pcCU->setIntraDirSubParts( CHANNEL_TYPE_LUMA,   bestPUMode[CHANNEL_TYPE_LUMA],   partOffset, depth + initTrDepth );
    pcCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, bestPUMode[CHANNEL_TYPE_CHROMA], partOffset, depth + initTrDepth );
    const UInt uiQPartNum = tuRecurseWithPU.GetAbsPartIdxNumParts();
    ::memcpy( pcCU->getTransformIdx() + partOffset, m_puhQTTempTrIdx, uiQPartNum * sizeof( UChar ) );
    ::memcpy( pcCU->getColourTransform() + partOffset, m_puhQTTempACTFlag, uiQPartNum * sizeof(Bool) );
    for (UInt component = 0; component < numberValidComponents; component++)
    {
      const ComponentID compID = ComponentID(component);
      ::memcpy( pcCU->getCbf( compID  )                           + partOffset, m_puhQTTempCbf[compID],                             uiQPartNum * sizeof( UChar ) );
      ::memcpy( pcCU->getTransformSkip( compID  )                 + partOffset, m_puhQTTempTransformSkipFlag[compID ],              uiQPartNum * sizeof( UChar ) );
      ::memcpy( pcCU->getCrossComponentPredictionAlpha(compID)    + partOffset, m_phQTTempCrossComponentPredictionAlpha[compID],    uiQPartNum * sizeof( SChar ) );
    }

    if( !tuRecurseWithPU.IsLastSection() )
    {
      assert( tuRecurseWithPU.ProcessChannelSection(CHANNEL_TYPE_CHROMA) );

      const UInt numChannelToProcess = getNumberValidComponents(pcCU->getPic()->getChromaFormat());

      for (UInt ch=0; ch < numChannelToProcess; ch++)
      {
        const ComponentID compID     = ComponentID(ch);
        const TComRectangle &puCRect = tuRecurseWithPU.getRect(compID);
        const UInt  compWidth        = puCRect.width;
        const UInt  compHeight       = puCRect.height;

        const UInt  ZOrder      = pcCU->getZorderIdxInCtu() + partOffset;
        Pel*  piDes             = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
        const UInt  desStride   = pcCU->getPic()->getPicYuvRec()->getStride( compID);
        const Pel*  piSrc       = pcRecoYuv->getAddr( compID, partOffset );
        const UInt  srcStride   = pcRecoYuv->getStride( compID);

        for( UInt y = 0; y < compHeight; y++, piSrc += srcStride, piDes += desStride )
        {
          for ( UInt x = 0; x < compWidth; x++ )
          {
            piDes[x] = piSrc[x];
          }
        }
      }
    }

  } while ( tuRecurseWithPU.nextSection(tuRecurseCU) );

  if( numPU > 1 )
  {
    UInt combCbfY = 0;
    UInt combCbfU = 0;
    UInt combCbfV = 0;
    UInt partIdx  = 0;

    for( UInt part = 0; part < 4; part++, partIdx += QNumParts )
    {
      combCbfY |= pcCU->getCbf( partIdx, COMPONENT_Y,  1 );
      combCbfU |= pcCU->getCbf( partIdx, COMPONENT_Cb, 1 );
      combCbfV |= pcCU->getCbf( partIdx, COMPONENT_Cr, 1 );
    }

    for( UInt offs = 0; offs < 4 * QNumParts; offs++ )
    {
      pcCU->getCbf( COMPONENT_Y  )[ offs ] |= combCbfY;
      pcCU->getCbf( COMPONENT_Cb )[ offs ] |= combCbfU;
      pcCU->getCbf( COMPONENT_Cr )[ offs ] |= combCbfV;
    }
  }

  pcCU->getTotalDistortion() = overallDistY + overallDistC;
}

Void
TEncSearch::estIntraPredLumaQTWithModeReuse(TComDataCU* pcCU,
                                            TComYuv*    pcOrgYuv,
                                            TComYuv*    pcPredYuv,
                                            TComYuv*    pcResiYuv,
                                            TComYuv*    pcRecoYuv,
                                            Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                           )
{
  const UInt         depth               = pcCU->getDepth(0);
  const UInt         initTrDepth         = pcCU->getPartitionSize(0) == SIZE_2Nx2N ? 0 : 1;
  const UInt         numPU               = 1<<(2*initTrDepth);
  const UInt         QNumParts           = pcCU->getTotalNumPart() >> 2;
  Distortion         overallDistY        = 0;
  Distortion         overallDistC        = 0;
  Pel                resiLumaPU[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];

  Bool    bMaintainResidual[NUMBER_OF_STORED_RESIDUAL_TYPES];
  for (UInt residualTypeIndex = 0; residualTypeIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; residualTypeIndex++)
  {
    bMaintainResidual[residualTypeIndex] = true; //assume true unless specified otherwise
  }

  bMaintainResidual[RESIDUAL_ENCODER_SIDE] = !(m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate());

  //===== set QP and clear Cbf =====
  if ( pcCU->getSlice()->getPPS()->getUseDQP() == true)
  {
    pcCU->setQPSubParts( pcCU->getQP(0), 0, depth );
  }
  else
  {
    pcCU->setQPSubParts( pcCU->getSlice()->getSliceQp(), 0, depth );
  }

  //===== loop over partitions =====
  TComTURecurse tuRecurseCU(pcCU, 0);
  TComTURecurse tuRecurseWithPU(tuRecurseCU, false, (initTrDepth==0)?TComTU::DONT_SPLIT : TComTU::QUAD_SPLIT);

  do
  {
    const UInt partOffset  = tuRecurseWithPU.GetAbsPartIdxTU();
    Distortion bestPUDistY = 0;
    Double     dBestPUCost  = 0;

    DEBUG_STRING_NEW(sMode)
    // set context models
    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[depth][CI_CURR_BEST] );

    xRecurIntraCodingLumaQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLumaPU, bestPUDistY, true, dBestPUCost, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sMode) );
    xSetIntraResultLumaQT( pcRecoYuv, tuRecurseWithPU );

    if (pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag())
    {
      const Int xOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).x0;
      const Int yOffset = tuRecurseWithPU.getRect( COMPONENT_Y ).y0;
      for (UInt storedResidualIndex = 0; storedResidualIndex < NUMBER_OF_STORED_RESIDUAL_TYPES; storedResidualIndex++)
      {
        if (bMaintainResidual[storedResidualIndex])
        {
          xStoreCrossComponentPredictionResult(resiLuma[storedResidualIndex], resiLumaPU[storedResidualIndex], tuRecurseWithPU, xOffset, yOffset, MAX_CU_SIZE, MAX_CU_SIZE );
        }
      }
    }

    overallDistY += bestPUDistY;

    //--- set reconstruction for next intra prediction blocks ---
    if( !tuRecurseWithPU.IsLastSection() )
    {
      const UInt numChannelToProcess = 1;

      for (UInt ch=0; ch<numChannelToProcess; ch++)
      {
        const ComponentID compID    = ComponentID(ch);
        const TComRectangle &puRect = tuRecurseWithPU.getRect(compID);
        const UInt  compWidth       = puRect.width;
        const UInt  compHeight      = puRect.height;

        const UInt  ZOrder      = pcCU->getZorderIdxInCtu() + partOffset;
        Pel*        piDes       = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
        const UInt  desStride   = pcCU->getPic()->getPicYuvRec()->getStride( compID);
        const Pel*  piSrc       = pcRecoYuv->getAddr( compID, partOffset );
        const UInt  srcStride   = pcRecoYuv->getStride( compID);

        for( UInt y = 0; y < compHeight; y++, piSrc += srcStride, piDes += desStride )
        {
          for( UInt x = 0; x < compWidth; x++ )
          {
            piDes[ x ] = piSrc[ x ];
          }
        }
      }
    }
  } while (tuRecurseWithPU.nextSection(tuRecurseCU));


  if( numPU > 1 )
  { // set Cbf for all blocks
    UInt combCbfY = 0;
    UInt combCbfU = 0;
    UInt combCbfV = 0;
    UInt partIdx  = 0;
    for( UInt part = 0; part < 4; part++, partIdx += QNumParts )
    {
      combCbfY |= pcCU->getCbf( partIdx, COMPONENT_Y,  1 );
      combCbfU |= pcCU->getCbf( partIdx, COMPONENT_Cb, 1 );
      combCbfV |= pcCU->getCbf( partIdx, COMPONENT_Cr, 1 );
    }
    for( UInt offs = 0; offs < 4 * QNumParts; offs++ )
    {
      pcCU->getCbf( COMPONENT_Y  )[ offs ] |= combCbfY;
      pcCU->getCbf( COMPONENT_Cb )[ offs ] |= combCbfU;
      pcCU->getCbf( COMPONENT_Cr )[ offs ] |= combCbfV;
    }
  }

  //===== reset context models =====
  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[depth][CI_CURR_BEST]);

  //===== set distortion (rate and r-d costs are determined later) =====
  assert(overallDistC == 0);
  pcCU->getTotalDistortion() = overallDistY + overallDistC;
}

Void
TEncSearch::estIntraPredChromaQTWithModeReuse(TComDataCU* pcCU,
                                              TComYuv*    pcOrgYuv,
                                              TComYuv*    pcPredYuv,
                                              TComYuv*    pcResiYuv,
                                              TComYuv*    pcRecoYuv,
                                              Pel         resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE]
                                             )
{
  const UInt    initTrDepth  = pcCU->getPartitionSize(0) != SIZE_2Nx2N && enable4ChromaPUsInIntraNxNCU(pcOrgYuv->getChromaFormat()) ? 1 : 0;

  TComTURecurse tuRecurseCU(pcCU, 0);
  TComTURecurse tuRecurseWithPU(tuRecurseCU, false, (initTrDepth==0)?TComTU::DONT_SPLIT : TComTU::QUAD_SPLIT);
  const UInt    QNumParts             = tuRecurseWithPU.GetAbsPartIdxNumParts();
  const UInt    depthCU               = tuRecurseWithPU.getCUDepth();
  const UInt    numberValidComponents = pcCU->getPic()->getNumberValidComponents();

  do
  {
    //----- init mode list -----
    if (tuRecurseWithPU.ProcessChannelSection(CHANNEL_TYPE_CHROMA))
    {
      const UInt  uiPartOffset   = tuRecurseWithPU.GetAbsPartIdxTU();

      assert( pcCU->getIntraDir( CHANNEL_TYPE_CHROMA, uiPartOffset ) == DM_CHROMA_IDX );  //YCgCo space only test DM_CHROMA_IDX

      //----- restore context models -----
      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[depthCU][CI_CURR_BEST] );

      DEBUG_STRING_NEW(sMode)
      //----- chroma coding -----
      Distortion dist = 0;
      xRecurIntraChromaCodingQT( pcOrgYuv, pcPredYuv, pcResiYuv, resiLuma, dist, tuRecurseWithPU DEBUG_STRING_PASS_INTO(sMode) );

      xSetIntraResultChromaQT( pcRecoYuv, tuRecurseWithPU );

      if( ! tuRecurseWithPU.IsLastSection() )
      {
        for (UInt ch = COMPONENT_Cb; ch < numberValidComponents; ch++)
        {
          const ComponentID compID    = ComponentID(ch);
          const TComRectangle &tuRect = tuRecurseWithPU.getRect(compID);
          const UInt  compWidth       = tuRect.width;
          const UInt  compHeight      = tuRect.height;
          const UInt  ZOrder          = pcCU->getZorderIdxInCtu() + tuRecurseWithPU.GetAbsPartIdxTU();
          Pel*  piDes                 = pcCU->getPic()->getPicYuvRec()->getAddr( compID, pcCU->getCtuRsAddr(), ZOrder );
          const UInt  desStride       = pcCU->getPic()->getPicYuvRec()->getStride( compID);
          const Pel*  piSrc           = pcRecoYuv->getAddr( compID, uiPartOffset );
          const UInt  srcStride       = pcRecoYuv->getStride( compID);

          for( UInt y = 0; y < compHeight; y++, piSrc += srcStride, piDes += desStride )
          {
            for( UInt x = 0; x < compWidth; x++ )
            {
              piDes[ x ] = piSrc[ x ];
            }
          }
        }
      }

      pcCU->getTotalDistortion() += dist;
    }

  } while (tuRecurseWithPU.nextSection(tuRecurseCU));

  //----- restore context models -----

  if( initTrDepth != 0 )
  { // set Cbf for all blocks
    UInt combCbfU = 0;
    UInt combCbfV = 0;
    UInt partIdx  = 0;
    for( UInt part = 0; part < 4; part++, partIdx += QNumParts )
    {
      combCbfU |= pcCU->getCbf( partIdx, COMPONENT_Cb, 1 );
      combCbfV |= pcCU->getCbf( partIdx, COMPONENT_Cr, 1 );
    }
    for( UInt offs = 0; offs < 4 * QNumParts; offs++ )
    {
      pcCU->getCbf( COMPONENT_Cb )[ offs ] |= combCbfU;
      pcCU->getCbf( COMPONENT_Cr )[ offs ] |= combCbfV;
    }
  }

  m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[depthCU][CI_CURR_BEST] );
}

UInt TEncSearch::paletteSearch(TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv*& rpcPredYuv, TComYuv*& rpcResiYuv, TComYuv*& rpcRecoYuv, Bool forcePalettePrediction, UInt iterNumber, UInt *paletteSizeCurrIter)
{
  UInt  depth      = pcCU->getDepth(0);
  Distortion  distortion = 0;
  Pel *paOrig[3], *paPalette[3];
  TCoeff *pRun;
  UChar *paSPoint[3];
  Pel *pPixelValue[3];
  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);
  UInt testMode=0;
  UInt paletteIdx = 0;
  for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch++)
  {
    paOrig[ch] = pcOrgYuv->getAddr((ComponentID)ch, 0);
    paPalette[ch] = pcCU->getPalette(ch, 0);
  }

  pRun = pcCU->getRun(COMPONENT_Y);
  paSPoint[0] = pcCU->getSPoint(COMPONENT_Y);
  UChar* pEscapeFlag = pcCU->getEscapeFlag(COMPONENT_Y);
  UInt paletteSize = 1;

  if (iterNumber < MAX_PALETTE_ITER)
  {
    if( pcCU->getCUTransquantBypass(0) )
    {
      xDerivePaletteLossless(pcCU, paPalette, paOrig, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize);
    }
    else if( forcePalettePrediction )
    {
      xDerivePaletteLossyForcePrediction(pcCU, paPalette, paOrig, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost);
    }
    else
    {
      xDerivePaletteLossy(pcCU, paPalette, paOrig, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost);
    }
  }
  else
  {
    paletteSize = m_prevPaletteSize[iterNumber - MAX_PALETTE_ITER];
    for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch++)
    {
      for ( paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
      {
        paPalette[ch][paletteIdx] = m_prevPalette[iterNumber - MAX_PALETTE_ITER][ch][paletteIdx];
      }
    }
    xDerivePaletteLossyIterative(pcCU, paPalette, paOrig, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost);
  }
  *paletteSizeCurrIter=paletteSize;

  pcCU->setPaletteSizeSubParts(0, paletteSize, 0, pcCU->getDepth(0));
  pcCU->setPaletteSizeSubParts(1, paletteSize, 0, pcCU->getDepth(0));
  pcCU->setPaletteSizeSubParts(2, paletteSize, 0, pcCU->getDepth(0));
  xReorderPalette(pcCU, paPalette, pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3);

  if (iterNumber==0)
  {
    m_forcePaletteSize=paletteSize;
    for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch++)
    {
      for ( paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
      {
        m_forcePalette[ch][paletteIdx]=paPalette[ch][paletteIdx];
      }
    }
  }
  else if (iterNumber==1)
  {
    UInt samePalette=0;
    if (paletteSize==m_forcePaletteSize)
    {
      samePalette=1;
      for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch++)
      {
        for ( paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
        {
          if (paPalette[ch][paletteIdx]!=m_forcePalette[ch][paletteIdx])
          {
            samePalette=0;
          }
        }
      }
    }
    if (samePalette)
    {
      pcCU->getTotalCost()=MAX_DOUBLE;
      return(0);
    }
  }

  UInt calcErroBits = iterNumber < MAX_PALETTE_ITER ? 1 : 0;
  xPreCalcPaletteIndexRD(pcCU, paPalette, paOrig, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost, calcErroBits);

  for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch++)
  {
    for ( paletteIdx = 0; paletteIdx < pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize(); paletteIdx++)
    {
      pcCU->setPaletteSubParts(ch,  paPalette[ch][paletteIdx], paletteIdx, 0, pcCU->getDepth(0));
    }
    pPixelValue[ch] = pcCU->getLevel(ComponentID (ch));
  }

  UInt width  = pcCU->getWidth(0);
  UInt height = pcCU->getHeight(0);
  UInt stride = rpcPredYuv->getStride(ComponentID(0));

  memcpy(m_paOriginalLevel, m_indexBlock, sizeof(Pel) * (width * height));

  UInt bits      = MAX_UINT;
  m_bBestScanRotationMode = 0;

  deriveRunAndCalcBits(pcCU, pcOrgYuv, rpcRecoYuv, bits, true,  PALETTE_SCAN_HORTRAV);
  if ( (pcCU->getPaletteSize( COMPONENT_Y, 0 ) + pcCU->getPaletteEscape( COMPONENT_Y, 0 )) > 1 )
  {
    deriveRunAndCalcBits( pcCU, pcOrgYuv, rpcRecoYuv, bits, false, PALETTE_SCAN_VERTRAV );
  }

  UInt errorOrig = 0, errorNew = 0;
  if (paletteSize > 2)
  {
    UInt totalPixel = width * height, uiTotalPixelC = (width>>scaleX) * (height>>scaleY);

    for (UInt ch = 0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
    {
      if ( ch == 0 )
      {
        memcpy(m_paLevelStoreRD[ch], m_paBestLevel[ch], sizeof(Pel) * totalPixel);
      }
      else
      {
        memcpy(m_paLevelStoreRD[ch], m_paBestLevel[ch], sizeof(Pel) * uiTotalPixelC);
      }
    }
    memcpy(m_paSPointStoreRD, m_paBestSPoint, sizeof(UChar) * totalPixel);
    memcpy(m_paRunStoreRD, m_paBestRun, sizeof(TCoeff) * totalPixel);
    memcpy(m_paEscapeFlagStoreRD, m_paBestEscapeFlag, sizeof(UChar) * totalPixel );

    memcpy(m_paletteInfoStoreRD, m_paletteInfoBest, sizeof(m_paletteInfo));

    m_paletteRunMode     = m_paBestSPoint;
    m_escapeFlagRD = m_paBestEscapeFlag;
    m_runRD        = m_paBestRun;
    m_levelRD[0]   = m_paBestLevel[0];
    m_levelRD[1]   = m_paBestLevel[1];
    m_levelRD[2]   = m_paBestLevel[2];

    preCalcRDMerge(pcCU, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost, &errorOrig, &errorNew, calcErroBits);
  }

  pcCU->setPaletteScanRotationModeFlagSubParts( m_bBestScanRotationMode, 0, depth );
  for (UInt ch = 0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
  {
    if ( ch == 0 )
    {
      memcpy( pPixelValue[ch], m_paBestLevel[ch], sizeof( Pel ) * width * height );
    }
    else
    {
      memcpy( pPixelValue[ch], m_paBestLevel[ch], sizeof( Pel ) * (width>>scaleX) * (height>>scaleY) );
    }
  }
  memcpy(paSPoint[0], m_paBestSPoint, sizeof(UChar) * width * height);
  memcpy(pRun,        m_paBestRun,    sizeof(TCoeff) * width * height);
  memcpy(pEscapeFlag, m_paBestEscapeFlag,  sizeof(UChar) * width * height);

  if (paletteSize > 2 )
  {
    UInt bitsRD;

    m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[depth][CI_CURR_BEST]);
    m_pcEntropyCoder->resetBits();
    xEncIntraHeader(pcCU, depth, 0, true, false);
    bitsRD = m_pcEntropyCoder->getNumberOfWrittenBits();

    Double rdOrig, rdNew;

    rdOrig = errorOrig + m_pcRdCost->getLambda()*(Double)bits;
    rdNew = errorNew + m_pcRdCost->getLambda()*(Double)bitsRD;

    if (rdNew <= rdOrig)
    {
      bits = bitsRD;
      m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[depth][CI_TEMP_BEST]);
    }
    else
    {
      for (UInt ch = 0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
      {
        if (ch == 0)
        {
          memcpy(pPixelValue[ch], m_paLevelStoreRD[ch], sizeof(Pel)* width * height);
        }
        else
        {
          memcpy(pPixelValue[ch], m_paLevelStoreRD[ch], sizeof(Pel)* (width >> scaleX) * (height >> scaleY));
        }
      }
      memcpy(paSPoint[0], m_paSPointStoreRD, sizeof(UChar)* width * height);
      memcpy(pRun, m_paRunStoreRD, sizeof(TCoeff)* width * height);
      memcpy(pEscapeFlag, m_paEscapeFlagStoreRD, sizeof(UChar)* width * height);
      memcpy(m_paletteInfoBest, m_paletteInfoStoreRD, sizeof(m_paletteInfo));
    }

    if (iterNumber<MAX_PALETTE_ITER) //after cabac loading fix
    {
      testMode = preCalcRD(pcCU, paPalette, pcCU->getWidth(0), pcCU->getHeight(0), paletteSize, m_pcRdCost, iterNumber);
    }
  }

  for (UInt ch = 0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
  {
    const ComponentID compID  = ComponentID(ch);
    width  = pcCU->getWidth(0)  >> pcCU->getPic()->getComponentScaleX(compID);
    height = pcCU->getHeight(0) >> pcCU->getPic()->getComponentScaleY(compID);
    stride = rpcPredYuv->getStride(compID);

    Pel *pOrig    = pcOrgYuv->getAddr  (compID, 0);
    Pel *pResi    = rpcResiYuv->getAddr(compID, 0);
    Pel *pPred    = rpcPredYuv->getAddr(compID, 0);
    Pel *pLevel   = pcCU->getLevel  (COMPONENT_Y);
    Pel *pPalette = pcCU->getPalette   (compID,0);
    Pel *pReco    = rpcRecoYuv->getAddr(compID, 0);
    Pel *pRecoPic = pcCU->getPic()->getPicYuvRec()->getAddr(ComponentID(ch), pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu());
    const UInt uiReconStride = pcCU->getPic()->getPicYuvRec()->getStride(ComponentID(ch));
    if(!m_bBestScanRotationMode)
    {
      UInt idx = 0;
      for( UInt y = 0; y < height; y++ )
      {
        for( UInt x = 0; x < width; x++ )
        {
          idx = (y << pcCU->getPic()->getComponentScaleY(compID)) * (width << pcCU->getPic()->getComponentScaleX(compID))
                  +(x << pcCU->getPic()->getComponentScaleX(compID));
          if(!pEscapeFlag[idx])
          {
            pPred[x] = pPalette[pLevel[idx]];
            pReco[x] = pPred[x];
          }
          pResi[x] = pOrig[x] - pPred[x];
          pRecoPic[x] = pReco[x];
        }

        pPred += stride;
        pResi += stride;
        pOrig += stride;
        pReco += stride;
        pRecoPic += uiReconStride;
      }
    }
    else
    {
      UInt idx = 0;
      for( UInt y = 0; y < width; y++ )
      {
        for( UInt x = 0; x < height; x++ )
        {
          idx = (y << pcCU->getPic()->getComponentScaleX(compID)) * (height << pcCU->getPic()->getComponentScaleY(compID))
                  + (x << pcCU->getPic()->getComponentScaleY(compID));
          UInt uiPxlPos = x*stride+y;
          if( !pEscapeFlag[idx] )
          {
            pPred[uiPxlPos] = pPalette[pLevel[idx]];
            pReco[uiPxlPos] = pPred[uiPxlPos];
          }
          pResi[uiPxlPos] = pOrig[uiPxlPos] - pPred[uiPxlPos];
          pRecoPic[x*uiReconStride+y] = pReco[uiPxlPos];
        }
      }
    }

  }

  for (UInt ch=0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
  {
    const ComponentID compID  = ComponentID(ch);
    const ChannelType chType = toChannelType(compID);
    width  = pcCU->getWidth(0)  >> pcCU->getPic()->getComponentScaleX(compID);
    height = pcCU->getHeight(0) >> pcCU->getPic()->getComponentScaleY(compID);
    stride = rpcPredYuv->getStride(compID);

    Pel *pOrig = pcOrgYuv->getAddr(compID, 0);
    Pel *pReco = rpcRecoYuv->getAddr(compID, 0);
    distortion += m_pcRdCost->getDistPart( pcCU->getSlice()->getSPS()->getBitDepth(chType), pReco, stride, pOrig, stride, width, height, compID );
  }

  Double dCost = m_pcRdCost->calcRdCost( bits, distortion );
  pcCU->getTotalBits()       = bits;
  pcCU->getTotalCost()       = dCost;
  pcCU->getTotalDistortion() = distortion;
  pcCU->copyToPic(depth);

  return (testMode);
}

Void TEncSearch::deriveRunAndCalcBits(TComDataCU* pcCU, TComYuv* pcOrgYuv, TComYuv* pcRecoYuv, UInt& minBits, Bool bReset, PaletteScanMode paletteScanMode)
{
  UInt depth = pcCU->getDepth(0);
  Pel *paOrig[3], *paLevel[3];
  TCoeff *pRun;
  UChar *paSPoint[3];
  Pel *pPixelValue[3];
  Pel * pRecoValue[3];

  const UInt width  = pcCU->getWidth(0);
  const UInt height = pcCU->getHeight(0);
  const UInt totalPixel = width * height;
  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);
  const UInt totalPixelC = totalPixel >> scaleX >> scaleY;

  paLevel[0] = pcCU->getLevel(COMPONENT_Y);
  pRun = pcCU->getRun(COMPONENT_Y);
  paSPoint[0] = pcCU->getSPoint(COMPONENT_Y);
  UChar *pEscapeFlag = pcCU->getEscapeFlag(COMPONENT_Y);

  for (UInt ch = 0; ch < 3; ch++)
  {
    paOrig[ch] = pcOrgYuv->getAddr((ComponentID)ch, 0);
    pPixelValue[ch] = pcCU->getLevel(ComponentID (ch));
    pRecoValue[ch] = pcRecoYuv->getAddr(ComponentID (ch), 0);
  }
  pcCU->setPaletteScanRotationModeFlagSubParts(paletteScanMode, 0, depth );
  if (paletteScanMode == PALETTE_SCAN_VERTRAV)
  {
    xRotationScan(m_indexBlock, width, height);
    xRotationScan(m_posBlock, width, height);
  }

  m_pScanOrder = g_scanOrder[SCAN_UNGROUPED][SCAN_TRAV][g_aucConvertToBit[width]+2][g_aucConvertToBit[height]+2];

  xDeriveRun(pcCU, paOrig, paLevel[0], paSPoint[0], pPixelValue, pRecoValue, pRun, width, height, pcOrgYuv->getStride(ComponentID(0)));

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[depth][CI_CURR_BEST]);
  m_pcEntropyCoder->resetBits();
  xEncIntraHeader ( pcCU, depth, 0, true, false);
  UInt tempBits = m_pcEntropyCoder->getNumberOfWrittenBits();
  if (minBits > tempBits)
  {
    m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[depth][CI_TEMP_BEST]);
    m_bBestScanRotationMode = paletteScanMode;
    memcpy(m_paletteInfoBest, m_paletteInfo, sizeof(m_paletteInfo));
    m_paletteNoElementsBest = m_paletteNoElements;

    memcpy(m_posBlockRD, m_posBlock, sizeof(m_posBlock));
    memcpy(m_indexBlockRD, m_indexBlock, sizeof(m_indexBlock));
    for (UInt ch = 0; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
    {
      if ( ch == 0 )
      {
        memcpy(m_paBestLevel[ch], pPixelValue[ch], sizeof(Pel) * totalPixel);
      }
      else
      {
        memcpy(m_paBestLevel[ch], pPixelValue[ch], sizeof(Pel) * totalPixelC);
      }
    }
    memcpy(m_paBestSPoint, paSPoint[0], sizeof(UChar) * totalPixel);
    memcpy(m_paBestRun, pRun, sizeof(TCoeff) * totalPixel);
    memcpy(m_paBestEscapeFlag, pEscapeFlag, sizeof(UChar) * totalPixel );
    minBits = tempBits;
  }
  if (bReset)
  {
    memcpy(m_indexBlock, m_paOriginalLevel, sizeof(Pel) * totalPixel);
    memset(paSPoint[0], 0, sizeof(UChar) * totalPixel);
    memset(pEscapeFlag, 0, sizeof(UChar) * totalPixel);
  }
}

UInt TEncSearch::preCalcRD(TComDataCU* pcCU, Pel *Palette[3], UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt iterNumber)
{
  UInt total = height * width;
  Bool bEscape = 0;
  UInt paletteSizeTemp=paletteSize, paletteSizeBest=paletteSize;
  UInt paletteIdxRemove1=0, paletteIdxRemove2=0, paletteIdxReplacement1 = MAX_PALETTE_SIZE-1,
       paletteIdxMapping1[MAX_PALETTE_SIZE] = { 0 }, paletteIdxMapping2[MAX_PALETTE_SIZE] = { 0 }, removedIndices[MAX_PALETTE_SIZE] = { 0 }, removedIndicesBest[MAX_PALETTE_SIZE] = { 0 };

  UInt64 error = 0;
  Double rdCostNew, rdCostOrig, rdCostDiff = MAX_DOUBLE;
  Int64  iBits=0, runBits=0;

  // Initial RD cost
  memset(removedIndices, 0, sizeof(removedIndices));

  for (UInt paletteIdx = 0; paletteIdx < MAX_PALETTE_SIZE; paletteIdx++)
  {
    paletteIdxMapping1[paletteIdx] = paletteIdx;
    paletteIdxMapping2[paletteIdx] = paletteIdx;
  }

  error = 0;
  for (UInt idx=0; idx < total; idx++)
  {
    if (m_indexBlockRD[idx] < 0)
    {
      m_indexBlockRD[idx] = (MAX_PALETTE_SIZE-1);
    }
    UInt curPaletteIdx = paletteIdxMapping1[m_indexBlockRD[idx]];
    error += m_indError[m_posBlockRD[idx]][curPaletteIdx];
  }


  for (UInt noElement = 0; noElement < m_paletteNoElementsBest; noElement++)
  {
    if (m_paletteInfoBest[noElement].paletteMode == PALETTE_RUN_LEFT && m_paletteInfoBest[noElement].index < (MAX_PALETTE_SIZE-1))
    {
      iBits += m_paletteInfoBest[noElement].bitsInd;
    }
  }

  rdCostOrig = pcCost->getLambda()*(Double)(iBits>>15) + error;

  // Initial error calculation
  Int64 errorDiffPaletteIndArray[MAX_PALETTE_SIZE][MAX_PALETTE_SIZE];
  Int64 bestIndToRemoveArray[MAX_PALETTE_SIZE][2];

  for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
  {
    memset(errorDiffPaletteIndArray[paletteIdx], 0, MAX_PALETTE_SIZE*sizeof(UInt64));
  }

  //Calculate all the SSE if index A is mapped to index B
  for (UInt idx=0; idx < total; idx++)
  {
    UInt uiOrgIdx = m_indexBlockRD[idx]; //Escape already converted
    UInt* indError = m_indError[m_posBlockRD[idx]];
    Int64* errDiffArray = errorDiffPaletteIndArray[uiOrgIdx];

    for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
    {
      errDiffArray[paletteIdx] += indError[paletteIdx];
    }
    errDiffArray[MAX_PALETTE_SIZE - 1] += indError[MAX_PALETTE_SIZE - 1];
  }

  //select the best mapped index for each index
  for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
  {
    Int64* bestIdxRemove = bestIndToRemoveArray[paletteIdx];
    Int64* errDiffArray = errorDiffPaletteIndArray[paletteIdx];

    bestIdxRemove[0] = MAX_PALETTE_SIZE-1;
    bestIdxRemove[1] = errDiffArray[MAX_PALETTE_SIZE-1] - errDiffArray[paletteIdx];

    for (UInt tmpIdx = 0; tmpIdx < paletteSize; tmpIdx++)
    {
      Int64 curError = errDiffArray[tmpIdx] - errDiffArray[paletteIdx];

      if (paletteIdx != tmpIdx && (curError < bestIdxRemove[1] ||
        (curError == bestIdxRemove[1] && tmpIdx < bestIdxRemove[0])))
      {
        bestIdxRemove[1] = curError;
        bestIdxRemove[0] = tmpIdx;
      }
    }
  }

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_CURR_BEST]);
  m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(1);

  //Remove the unused index after the mapping
  while (paletteSizeTemp > 2)
  {
    Int64 minError = MAX_INT64;
    UInt paletteCnt = 0;
    while (paletteCnt < paletteSize)
    {
      if (removedIndices[paletteCnt] == 0)
      {
        if (bestIndToRemoveArray[paletteCnt][1] < minError)
        {
          minError           = bestIndToRemoveArray[paletteCnt][1];
          paletteIdxRemove1      = paletteCnt;
          paletteIdxReplacement1 = UInt(bestIndToRemoveArray[paletteCnt][0]);
        }
      }
      paletteCnt++;
    }

    // Merge removed index paletteIdxRemove1 with paletteIdxReplacement1
    if (paletteIdxReplacement1 != (MAX_PALETTE_SIZE-1))
    {
      for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
      {
        if (removedIndices[paletteIdx] == 0)
        {
          errorDiffPaletteIndArray[paletteIdxReplacement1][paletteIdx] += errorDiffPaletteIndArray[paletteIdxRemove1][paletteIdx];
        }
      }
      errorDiffPaletteIndArray[paletteIdxReplacement1][MAX_PALETTE_SIZE-1] += errorDiffPaletteIndArray[paletteIdxRemove1][MAX_PALETTE_SIZE-1];
    }
    removedIndices[paletteIdxRemove1]=1;

    // Find another min index for paletteIdxReplacement1 and indices for which paletteIdxRemove1 was chosen
    for (UInt paletteIdxOrg = 0; paletteIdxOrg < paletteSize; paletteIdxOrg++)
    {
      if ((removedIndices[paletteIdxOrg] == 0 && bestIndToRemoveArray[paletteIdxOrg][0] == paletteIdxRemove1) || paletteIdxOrg == paletteIdxReplacement1)
      {
        Int64* bestIdxRemove = bestIndToRemoveArray[paletteIdxOrg];
        Int64* errDiffArray = errorDiffPaletteIndArray[paletteIdxOrg];

        bestIdxRemove[0] = MAX_PALETTE_SIZE-1;
        bestIdxRemove[1] = errDiffArray[MAX_PALETTE_SIZE - 1] - errDiffArray[paletteIdxOrg];

        for (UInt paletteIdxTmp = 0; paletteIdxTmp < paletteSize; paletteIdxTmp++)
        {
          if (removedIndices[paletteIdxTmp] == 0)
          {
            Int64 curError = errDiffArray[paletteIdxTmp] - errDiffArray[paletteIdxOrg];

            if (paletteIdxTmp != paletteIdxOrg && (curError < bestIdxRemove[1] ||
              (curError == bestIdxRemove[1] && paletteIdxTmp<bestIdxRemove[0])))
            {
              bestIndToRemoveArray[paletteIdxOrg][1] = curError;
              bestIndToRemoveArray[paletteIdxOrg][0] = paletteIdxTmp;
            }
          }
        }
      }
    }

    paletteSizeTemp--;

    // Remapping
    for (UInt paletteIdx = 0; paletteIdx<paletteSize; paletteIdx++)
    {
      if (paletteIdx == paletteIdxRemove1)
      {
        removedIndices[paletteIdx] = 1;
      }
      if (paletteIdxMapping1[paletteIdx] == paletteIdxRemove1)
      {
        paletteIdxMapping1[paletteIdx] = paletteIdxReplacement1;
      }
    }

    paletteIdxRemove2 = paletteIdxMapping2[paletteIdxRemove1];
    UInt paletteIdxReplacement2 = paletteIdxMapping2[paletteIdxReplacement1];

    for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
    {
      if (paletteIdxMapping2[paletteIdx] == paletteIdxRemove2)
      {
        paletteIdxMapping2[paletteIdx] = paletteIdxReplacement2;
      }
      if (paletteIdxMapping2[paletteIdx] > paletteIdxRemove2 && paletteIdxMapping2[paletteIdx] < (MAX_PALETTE_SIZE - 1))
      {
        paletteIdxMapping2[paletteIdx]--;
      }
    }

    if (paletteIdxReplacement1 == (MAX_PALETTE_SIZE-1))
    {
      bEscape = 1;
    }
    UInt uiIndexMaxSize = paletteSizeTemp;
    if (pcCU->getPaletteEscape(COMPONENT_Y, 0) || bEscape == 1)
    {
      uiIndexMaxSize++;
    }

    // New RD cost
    error=0;
    for (UInt idx = 0; idx < total; idx++)
    {
      UInt curPaletteIdx = paletteIdxMapping1[m_indexBlockRD[idx]];
      error += m_indError[m_posBlockRD[idx]][curPaletteIdx];
    }

    UInt currIndex, nextIndex, noSameIndices, predIndex, run=0;

    iBits = 0;
    UInt noElement = 0;
    while (noElement < m_paletteNoElementsBest)
    {
      noSameIndices = 0;

      currIndex = paletteIdxMapping2[m_paletteInfoBest[noElement].index];
      run = m_paletteInfoBest[noElement].run;
      runBits = m_paletteInfoBest[noElement].bitsRun;

      if (m_paletteInfoBest[noElement].paletteMode == PALETTE_RUN_LEFT && currIndex < (MAX_PALETTE_SIZE-1))
      {
        UInt idx=1;
        while((noElement + idx) < m_paletteNoElementsBest)
        {
          nextIndex = paletteIdxMapping2[m_paletteInfoBest[noElement+1].index];
          if (m_paletteInfoBest[noElement+idx].paletteMode == PALETTE_RUN_LEFT && nextIndex < (MAX_PALETTE_SIZE-1) && nextIndex == currIndex)
          {
            run     += (m_paletteInfoBest[noElement+idx].run+1);
            runBits += m_paletteInfoBest[noElement+idx].bitsRun;
            idx++;
          }
          else
          {
            break;
          }
        }
        noSameIndices = idx-1;

        if (noElement > 0)
        {
          predIndex = paletteIdxMapping2[m_paletteInfoBest[noElement].indexPred];
          if (currIndex >= predIndex && currIndex > 0)
          {
            currIndex--;
          }
          iBits += xGetTruncBinBits(currIndex, uiIndexMaxSize-1) << 15;
        }
        else
        {
          iBits += xGetTruncBinBits(currIndex, uiIndexMaxSize) << 15;
        }

        if (noSameIndices>0)
        {
          m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
          m_pcRDGoOnSbacCoder->resetBits();
          UInt64 initialBits = m_pcRDGoOnSbacCoder->getNumPartialBits();
          m_pcRDGoOnSbacCoder->encodeRun(run, PALETTE_RUN_LEFT, currIndex, total - m_paletteInfoBest[noElement].position - 1);
          iBits += (m_pcRDGoOnSbacCoder->getNumPartialBits() - runBits - initialBits);
        }
      }
      noElement += (1 + noSameIndices);
    }

    rdCostNew = pcCost->getLambda()*(Double)(iBits>>15)+error;

    if ((rdCostNew - rdCostOrig) < rdCostDiff)
    {
      rdCostDiff = (rdCostNew-rdCostOrig);
      paletteSizeBest = paletteSizeTemp;
      memcpy(removedIndicesBest, removedIndices, sizeof(removedIndices));
    }
  }

  UInt testReducedInd=1;

  m_prevPaletteSize[iterNumber]=paletteSizeBest;

  UInt paletteCnt=0;
  for (UInt paletteIdx = 0; paletteIdx < paletteSize; paletteIdx++)
  {
    if (removedIndicesBest[paletteIdx] == 0)
    {
      for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1: 3); ch++)
      {
        m_prevPalette[iterNumber][ch][paletteCnt] = Palette[ch][paletteIdx];
      }
      paletteCnt++;
    }
  }

  return(testReducedInd);
}

UInt TEncSearch::calcPaletteIndexPredAndBits(Int iMaxSymbol, UInt idxStart, UInt width, UInt *predIndex, UInt *currIndex)
{
  UInt traIdx, indexBits;

  *predIndex=iMaxSymbol - 1;
  if(idxStart)
  {
    traIdx=m_pScanOrder[idxStart];
    UInt traIdxLeft = m_pScanOrder[idxStart - 1];
    if (m_paletteRunMode[traIdxLeft] == PALETTE_RUN_LEFT)  ///< copy left
    {
      *predIndex = m_indexBlockRD[traIdxLeft];
      if(m_indexBlockRD[traIdxLeft]==(MAX_PALETTE_SIZE-1))
      {
        *predIndex = iMaxSymbol - 1;
      }
    }
    else
    {
      *predIndex = m_indexBlockRD[traIdx - width];
      if(m_indexBlockRD[traIdx - width]==(MAX_PALETTE_SIZE-1))
      {
        *predIndex = iMaxSymbol - 1;
      }
    }
    iMaxSymbol--;
  }

  if ((*currIndex)>(*predIndex))
  {
    (*currIndex)--;
  }
  indexBits = xGetTruncBinBits(*currIndex, iMaxSymbol);

  return(indexBits);
}

UInt TEncSearch::calcPaletteStartCopy(UInt positionInit, UInt positionCurrSegment, UInt width)
{
  UInt positionStart;

  positionStart=positionInit;
  while (positionStart>(positionCurrSegment+1) && positionStart>width)
  {

    UInt traIdx = m_pScanOrder[positionStart-1];
    if (m_indexBlockRD[traIdx]==m_indexBlockRD[traIdx - width])
    {
      positionStart--;
    }
    else
    {
      break;
    }
  }

  return(positionStart);
}

UInt TEncSearch::calcPaletteErrorCopy(UInt idxStart, UInt run, UInt width, UInt *merge)
{
  UInt error=0;
  UInt idx, traIdx, paletteIdx;
  *merge=1;

  for (idx=idxStart; idx<=(idxStart+run); idx++)
  {
    traIdx = m_pScanOrder[idx];

    paletteIdx=m_indexBlockRD[traIdx - width];
    if (paletteIdx==(MAX_PALETTE_SIZE-1))
    {
      *merge=0;
      break;
    }
    error+=m_indError[m_posBlockRD[traIdx]][paletteIdx];
  }

  return(error);
}

UInt64 TEncSearch::calcPaletteErrorLevel(Int idxStart, UInt run, UInt paletteIdx)
{
  UInt64 error = 0;

  for (Int idx = idxStart + run; idx >= idxStart; idx--)
  {
    error += m_indError[m_posBlockRD[m_pScanOrder[idx]]][paletteIdx];
  }

  return(error);
}

Void TEncSearch::modifyPaletteSegment(UInt width, UInt idxStart, UInt paletteMode, UInt paletteIdx, UInt run)
{
  for (UInt idx=idxStart; idx<=(idxStart+run); idx++)
  {
    UInt uiTraIdx = m_pScanOrder[idx];

    m_paletteRunMode[uiTraIdx]=paletteMode;

    if (paletteMode==PALETTE_RUN_LEFT)
    {
      m_indexBlockRD[uiTraIdx]=paletteIdx;
    }
    else
    {
      m_indexBlockRD[uiTraIdx]=m_indexBlockRD[uiTraIdx - width];
    }
  }
}

UInt TEncSearch::findPaletteSegment(PaletteInfoStruct *paletteElement, UInt idxStart, UInt indexMaxSize, UInt width, UInt total, UInt copyPixels[],
  Int restrictLevelRun, UInt calcErrBits)
{
  UInt traIdxStart=m_pScanOrder[idxStart], idx, traIdx, paletteMode, predIndex=0, run, currIndex=0;
  UInt64 indexBits=0, runBits, sPointBits;
  Int iMaxSymbol;
  if (m_paletteRunMode[traIdxStart]==PALETTE_RUN_LEFT)
  {
    paletteMode=PALETTE_RUN_LEFT;
    iMaxSymbol=indexMaxSize;

    currIndex=m_indexBlockRD[traIdxStart];

    UInt startIndex = currIndex;

    if(m_indexBlockRD[traIdxStart]==(MAX_PALETTE_SIZE-1))
    {
      currIndex = iMaxSymbol - 1;
      paletteElement->index=(MAX_PALETTE_SIZE-1);
    }
    else
    {
      paletteElement->index=currIndex;
    }

    if (restrictLevelRun==-1)
    {
      run=0;
      idx=idxStart;
      while (idx<(total-1))
      {
        idx++;
        traIdx=m_pScanOrder[idx];

        if (m_indexBlockRD[traIdx] == startIndex)
        {
          run++;
        }
        else
        {
          break;
        }
      }
    }
    else if (restrictLevelRun==-2)
    {
      run=0;
      idx=idxStart;
      while (idx<(total-1))
      {
        idx++;
        traIdx=m_pScanOrder[idx];

        if (m_indexBlockRD[traIdx] == startIndex && m_paletteRunMode[traIdx] == PALETTE_RUN_LEFT)
        {
          run++;
        }
        else
        {
          break;
        }
      }
    }
    else
    {
      run=restrictLevelRun;
    }
    if (calcErrBits)
    {
      indexBits=calcPaletteIndexPredAndBits(iMaxSymbol, idxStart, width, &predIndex, &currIndex);
    }
  }
  else
  {

    paletteMode = PALETTE_RUN_ABOVE;
    paletteElement->index = 0;

    run = 0;
    idx = idxStart;
    traIdx = m_pScanOrder[idx];
    copyPixels[traIdx - width] = 1;

    m_indexBlockRD[traIdx] = m_indexBlockRD[traIdx - width];
    while (idx < (total-1))
    {
      idx++;
      traIdx=m_pScanOrder[idx];
      if (m_indexBlockRD[traIdx]==m_indexBlockRD[traIdx - width])
      {
        copyPixels[traIdx - width]=1;
        run++;
      }
      else
      {
        break;
      }
    }
  }

  paletteElement->position = idxStart;
  paletteElement->paletteMode = paletteMode;
  paletteElement->run = run;

  for (idx = idxStart; idx <= (idxStart + run); idx++)
  {
    traIdx = m_pScanOrder[idx];

    m_paletteRunMode[traIdx] = paletteMode;
    m_runRD[traIdx] = run;

    if (m_escapeFlagRD[traIdx] == 0)
    {
      m_levelRD[0][traIdx] = m_indexBlockRD[traIdx];
    }
  }

  if (calcErrBits)
  {

    UInt escape = 0, usedForCopy = 0;
    UInt64 error = 0;

    for (idx=idxStart; idx<=(idxStart+run); idx++)
    {
      traIdx = m_pScanOrder[idx];
      Pel index = m_indexBlockRD[traIdx];
      error += m_indError[m_posBlockRD[traIdx]][index];
      if (copyPixels[traIdx])
      {
        usedForCopy = 1;
      }

      UInt escFlagOrig=m_escapeFlagRD[traIdx];
      UInt escFlagNew = index == (MAX_PALETTE_SIZE - 1) ? 1 : 0;
      assert(escFlagOrig == escFlagNew);

      if (escFlagNew == 1)
      {
        escape = 1;
      }
    }

    m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
    m_pcRDGoOnSbacCoder->resetBits();
    UInt64 initialBits = m_pcRDGoOnSbacCoder->getNumPartialBits();
    m_pcRDGoOnSbacCoder->encodeSPointRD(idxStart, width, m_paletteRunMode, paletteMode, m_pScanOrder);
    sPointBits=m_pcRDGoOnSbacCoder->getNumPartialBits()-initialBits;
    m_pcRDGoOnSbacCoder->encodeRun(run, paletteMode, currIndex, total - idxStart - 1);


    runBits=m_pcRDGoOnSbacCoder->getNumPartialBits()-sPointBits-initialBits;


    paletteElement->position = idxStart;
    paletteElement->paletteMode = paletteMode;
    paletteElement->run=run;
    paletteElement->indexPred = predIndex;
    paletteElement->bitsInd=indexBits<<15;
    paletteElement->bitsRun = runBits;
    paletteElement->bitsAll=runBits+sPointBits+(indexBits<<15);
    paletteElement->error = Double(error);
    paletteElement->escape = escape;
    paletteElement->usedForCopy = usedForCopy;
  }

  return(run);
}

Void TEncSearch::preCalcRDMerge(TComDataCU* pcCU, UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt *errorOrig, UInt *errorNew, UInt calcErrBits)
{
  UInt idx, idxStart, traIdx, noElement, run, total = height * width,
     merge, forceMerge;

  UInt predIndex = 0;
  Double error = 0;

  Double rdCostOrig, rdCostBestMode;
  UInt copyPixels[32*32];
  Int modMode;

  UInt modRunMode, modRunCurrentBest=0, uiModPositionNextBest=0, modRunNextBest=0;
  Double rdCostModRun, rdCostModRunMin = MAX_DOUBLE;
  UInt paletteMode=PALETTE_RUN_LEFT, paletteModeMerge=PALETTE_RUN_LEFT, idxStartMerge, mergeCurrIndex=0, currIndex, mergePaletteIdx=0;
  UInt64 indexBits;
  Double rdCostMerge=MAX_DOUBLE, errorMin = MAX_DOUBLE;
  Int iMaxSymbol;

  UInt paletteIdx = 0;
  UInt paletteIdxStart, paletteIdxEnd;

  PaletteInfoStruct* currentPaletteElement = &m_currentPaletteElement;
  PaletteInfoStruct* nextPaletteElement = &m_nextPaletteElement;
  PaletteInfoStruct *tempPaletteElement;

  m_pScanOrder = g_scanOrder[SCAN_UNGROUPED][SCAN_TRAV][g_aucConvertToBit[width]+2][g_aucConvertToBit[height]+2];

  UInt64 errOrig = 0;

  for (idx=0; idx<total; idx++)
  {
    traIdx = m_pScanOrder[idx];
    if (m_escapeFlagRD[traIdx])
    {
      errOrig += m_indError[m_posBlockRD[traIdx]][MAX_PALETTE_SIZE];
    }
    else
    {
      paletteIdx = m_indexBlockRD[traIdx];
      errOrig += m_indError[m_posBlockRD[traIdx]][paletteIdx];
    }
  }

  *errorOrig = UInt(errOrig);

  UInt indexMaxSize = paletteSize;
  if( pcCU->getPaletteEscape(COMPONENT_Y, 0) )
  {
    indexMaxSize++;
  }

  memset(copyPixels, 0, sizeof(copyPixels));

  for (noElement=0; noElement<m_paletteNoElementsBest; noElement++)
  {
    if (m_paletteInfoBest[noElement].paletteMode==PALETTE_RUN_ABOVE)
    {
      UInt end = m_paletteInfoBest[noElement].position + m_paletteInfoBest[noElement].run;
      for (idx = m_paletteInfoBest[noElement].position; idx <= end; idx++)
      {
        traIdx = m_pScanOrder[idx];
        copyPixels[traIdx - width]=1;
      }
    }
  }

  for (idx=0; idx<total; idx++)
  {
    if (m_indexBlockRD[idx]<0)
    {
      m_indexBlockRD[idx]=(MAX_PALETTE_SIZE-1);
    }
  }

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_CURR_BEST]);
  m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(1);

  idxStart=0;
  findPaletteSegment(currentPaletteElement, idxStart, indexMaxSize, width, total, copyPixels, -1, 1);
  idxStart=currentPaletteElement->position+(currentPaletteElement->run+1);

  while (idxStart < total)
  {
    findPaletteSegment(nextPaletteElement, idxStart, indexMaxSize, width, total, copyPixels, -1, 1);
    modMode = 0; merge = 0; forceMerge = 0;
    if (currentPaletteElement->escape == 0 && nextPaletteElement->escape == 0)
    {
      run=currentPaletteElement->run + nextPaletteElement->run + 1;

      rdCostOrig=pcCost->getLambda()*(Double)((currentPaletteElement->bitsAll+nextPaletteElement->bitsAll)>>15)+
        currentPaletteElement->error+nextPaletteElement->error;
      rdCostBestMode = rdCostOrig;

      if (currentPaletteElement->paletteMode == nextPaletteElement->paletteMode && (currentPaletteElement->paletteMode == PALETTE_RUN_ABOVE || currentPaletteElement->index == nextPaletteElement->index))
      {
        forceMerge = 1;
      }

      modRunMode=0;
      if (currentPaletteElement->paletteMode==PALETTE_RUN_LEFT && nextPaletteElement->paletteMode==PALETTE_RUN_ABOVE && forceMerge==0)
      {
        UInt modPositionNext, runCurrent, runNext, tempIndex, groupInit, groupNew;
        UInt64 modRunBits;

        rdCostModRunMin=MAX_DOUBLE;
        Int startCopy=calcPaletteStartCopy(nextPaletteElement->position, currentPaletteElement->position, width);
        runCurrent=startCopy-currentPaletteElement->position-1;
        runNext=run-runCurrent-1;

        groupInit=m_runGolombGroups[runNext];

        for (modPositionNext=startCopy; modPositionNext<nextPaletteElement->position; modPositionNext++)
        {
          runCurrent=modPositionNext-currentPaletteElement->position-1;
          runNext=run-runCurrent-1;

          groupNew=m_runGolombGroups[runNext];

          if (modPositionNext==startCopy || groupNew<groupInit)
          {
            groupInit=(groupNew<groupInit)? groupNew : groupInit;
            modRunMode=1;

            m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
            m_pcRDGoOnSbacCoder->resetBits();
            UInt64 initialBits=m_pcRDGoOnSbacCoder->getNumPartialBits();
            m_pcRDGoOnSbacCoder->encodeSPointRD(currentPaletteElement->position, width, m_paletteRunMode, PALETTE_RUN_LEFT, m_pScanOrder);
            tempIndex=currentPaletteElement->index > currentPaletteElement->indexPred ? currentPaletteElement->index-1 : currentPaletteElement->index ;
            m_pcRDGoOnSbacCoder->encodeRun(runCurrent, PALETTE_RUN_LEFT, tempIndex, total - currentPaletteElement->position - 1);

            // Next palette segment
            m_pcRDGoOnSbacCoder->encodeSPointRD(modPositionNext, width, m_paletteRunMode, PALETTE_RUN_ABOVE, m_pScanOrder);

            m_pcRDGoOnSbacCoder->encodeRun(runNext, PALETTE_RUN_ABOVE, tempIndex, total - modPositionNext - 1);

            modRunBits=m_pcRDGoOnSbacCoder->getNumPartialBits()+currentPaletteElement->bitsInd-initialBits;
            rdCostModRun=pcCost->getLambda()*(Double)(modRunBits>>15)+currentPaletteElement->error+nextPaletteElement->error;


            if (rdCostModRun<rdCostModRunMin)
            {
              rdCostModRunMin=rdCostModRun;
              uiModPositionNextBest=modPositionNext;
              modRunCurrentBest=runCurrent;
              modRunNextBest=runNext;
            }
          }
        }
      }

      if (modRunMode==1 && rdCostModRunMin<=rdCostBestMode)
      {
        rdCostBestMode=rdCostModRunMin;
        modMode=1;
      }

      if( !pcCU->getCUTransquantBypass(0) || pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_444 )
      {
        idxStartMerge = currentPaletteElement->position;
        predIndex = currentPaletteElement->indexPred;

        UInt testLevel=0;
        if (currentPaletteElement->paletteMode==PALETTE_RUN_LEFT)
        {
          if (currentPaletteElement->paletteMode==PALETTE_RUN_LEFT && nextPaletteElement->paletteMode==PALETTE_RUN_LEFT && (currentPaletteElement->usedForCopy==0 || nextPaletteElement->usedForCopy==0))
          {
            testLevel=1;
          }
          if (nextPaletteElement->paletteMode==PALETTE_RUN_ABOVE && nextPaletteElement->usedForCopy==0)
          {
            testLevel=1;
          }
          if (forceMerge==1)
          {
            testLevel=1;
          }
        }

        UInt testCopy=(currentPaletteElement->position>=width) ? 1 : 0;
        if (currentPaletteElement->paletteMode==PALETTE_RUN_LEFT && currentPaletteElement->usedForCopy==1)
        {
          testCopy=0;
        }
        if (nextPaletteElement->paletteMode==PALETTE_RUN_LEFT && nextPaletteElement->usedForCopy==1)
        {
          testCopy=0;
        }

        if (testLevel)
        {
          paletteMode = PALETTE_RUN_LEFT;
          iMaxSymbol = (idxStartMerge > 0) ? indexMaxSize - 1 : indexMaxSize;
          paletteIdxStart = 0, paletteIdxEnd = paletteSize - 1;

          if (currentPaletteElement->usedForCopy == 1)
          {
            paletteIdxStart = currentPaletteElement->index;
            paletteIdxEnd = currentPaletteElement->index;
          }
          else if (nextPaletteElement->usedForCopy == 1)
          {
            paletteIdxStart = nextPaletteElement->index;
            paletteIdxEnd = nextPaletteElement->index;
          }

          errorMin = MAX_DOUBLE;
          for (paletteIdx = paletteIdxStart; paletteIdx <= paletteIdxEnd; paletteIdx++)
          {
            if (paletteIdx != predIndex)
            { // do not allow copy mode
              merge = 1;
              currIndex = paletteIdx > predIndex ? paletteIdx - 1 : paletteIdx;

              indexBits = xGetTruncBinBits(currIndex, iMaxSymbol);
              error = pcCost->getLambda()*indexBits; // error includes index
              error += calcPaletteErrorLevel(idxStartMerge, run, paletteIdx);

              if (error<errorMin)
              {
                errorMin = error;
                mergePaletteIdx = paletteIdx;
                mergeCurrIndex = currIndex;
              }
            }
          }

          if (merge==1)
          {
            m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
            m_pcRDGoOnSbacCoder->resetBits();
            UInt64 initialBits=m_pcRDGoOnSbacCoder->getNumPartialBits();
            m_pcRDGoOnSbacCoder->encodeSPointRD(idxStartMerge, width, m_paletteRunMode, paletteMode, m_pScanOrder);
            m_pcRDGoOnSbacCoder->encodeRun(run, paletteMode, mergeCurrIndex, total - idxStartMerge - 1);

            rdCostMerge=pcCost->getLambda()*(Double)((m_pcRDGoOnSbacCoder->getNumPartialBits()-initialBits)>>15)+errorMin;
          }

          if ((merge==1 && rdCostMerge<=rdCostBestMode) || forceMerge==1)
          {
            rdCostBestMode=rdCostMerge;
            modMode=2;
            paletteModeMerge=PALETTE_RUN_LEFT;
          }
        }

        if (testCopy)
        {
          paletteMode=PALETTE_RUN_ABOVE;
          errorMin=calcPaletteErrorCopy(idxStartMerge, run, width, &merge);

          if (currentPaletteElement->paletteMode==PALETTE_RUN_ABOVE && nextPaletteElement->paletteMode==PALETTE_RUN_ABOVE)
          {
            merge=1;
          }

          if (merge == 1)
          {
            m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
            m_pcRDGoOnSbacCoder->resetBits();
            UInt64 initialBits = m_pcRDGoOnSbacCoder->getNumPartialBits();
            m_pcRDGoOnSbacCoder->encodeSPointRD(idxStartMerge, width, m_paletteRunMode, paletteMode, m_pScanOrder);
            m_pcRDGoOnSbacCoder->encodeRun(run, paletteMode, mergeCurrIndex, total - idxStartMerge - 1);
            rdCostMerge = pcCost->getLambda()*(Double)((m_pcRDGoOnSbacCoder->getNumPartialBits() - initialBits) >> 15) + errorMin;
          }

          if (merge==1 && rdCostMerge<=rdCostBestMode)
          {
            rdCostBestMode=rdCostMerge;
            modMode=2;
            paletteModeMerge=PALETTE_RUN_ABOVE;
          }
        }
      }
    }

    if (modMode==0)
    {
      tempPaletteElement=currentPaletteElement;
      currentPaletteElement=nextPaletteElement;
      nextPaletteElement=tempPaletteElement;
    }
    else
    {
      if (modMode==1)
      {
        modifyPaletteSegment(width, uiModPositionNextBest, nextPaletteElement->paletteMode, nextPaletteElement->index, modRunNextBest);
        findPaletteSegment(currentPaletteElement, currentPaletteElement->position, indexMaxSize, width, total, copyPixels, modRunCurrentBest, 1);
        findPaletteSegment(currentPaletteElement, uiModPositionNextBest, indexMaxSize, width, total, copyPixels, -1, 1);
      }
      if (modMode==2)
      {
        modifyPaletteSegment(width, currentPaletteElement->position, paletteModeMerge, mergePaletteIdx, currentPaletteElement->run);
        modifyPaletteSegment(width, nextPaletteElement->position, paletteModeMerge, mergePaletteIdx, nextPaletteElement->run);
        findPaletteSegment(currentPaletteElement, currentPaletteElement->position, indexMaxSize, width, total, copyPixels, -1, 1);
      }
    }

    idxStart=currentPaletteElement->position+(currentPaletteElement->run+1);
  }

  idxStart=0; noElement=0;
  while (idxStart<total)
  {
    findPaletteSegment(m_paletteInfoBest + noElement, idxStart, indexMaxSize, width, total, copyPixels, -2, calcErrBits);
    idxStart=m_paletteInfoBest[noElement].position+(m_paletteInfoBest[noElement].run+1);
    noElement++;
  }

  m_paletteNoElementsBest=noElement;

  UInt64 errNew = 0;

  for (idx=0; idx<total; idx++)
  {
    traIdx = m_pScanOrder[idx];
    if (m_escapeFlagRD[traIdx])
    {
      errNew += m_indError[m_posBlockRD[traIdx]][MAX_PALETTE_SIZE];
    }
    else
    {
      paletteIdx=m_indexBlockRD[traIdx];
      errNew += m_indError[m_posBlockRD[traIdx]][paletteIdx];
    }
  }

  *errorNew = UInt(errNew);
}

Void TEncSearch::xDeriveRun(TComDataCU* pcCU, Pel* pOrg[3], Pel* pValue, UChar* pSPoint,
  Pel** paPixelValue, Pel ** paRecoValue, TCoeff* pRun,
  UInt width, UInt height,  UInt strideOrg)
{
  UInt total = height * width, idx = 0;
  UInt startPos = 0,  run = 0, copyRun = 0;
  Int temp = 0;
  UInt traIdx;  //unified position variable (raster scan)
  Pel *pcIndexBlock = m_indexBlock;

  UChar *pEscapeFlag  = pcCU->getEscapeFlag(COMPONENT_Y);
  UInt noElements=0;
  UInt64 allBitsCopy = 0, allBitsIndex = 0, indexBits = 0, runBitsIndex = 0, runBitsCopy = 0;

  UInt indexMaxSize = pcCU->getPaletteSize(COMPONENT_Y, 0);
  if( pcCU->getPaletteEscape(COMPONENT_Y, 0) )
  {
    indexMaxSize++;
  }

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[pcCU->getDepth(0)][CI_CURR_BEST]);
  m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(1);

  //Test Run
  while (idx < total)
  {
    startPos = idx;
    Double dAveBitsPerPix[NUM_PALETTE_RUN];

    run = 0;
    Bool RunValid = xCalLeftRun(pcCU, pValue, pSPoint, startPos, total, run, pEscapeFlag);

    if(RunValid)
    {
      dAveBitsPerPix[PALETTE_RUN_LEFT] = xGetRunBits(pcCU, pValue, startPos, (run + 1), PALETTE_RUN_LEFT, &allBitsIndex, &indexBits, &runBitsIndex);
    }
    else
    {
      dAveBitsPerPix[PALETTE_RUN_LEFT] = std::numeric_limits<double>::max();
    }

    copyRun = 0;
    Bool CopyValid = xCalAboveRun(pcCU, pValue, pSPoint, width, startPos, total, copyRun, pEscapeFlag);

    if(CopyValid)
    {
      dAveBitsPerPix[PALETTE_RUN_ABOVE] = xGetRunBits(pcCU, pValue, startPos, copyRun, PALETTE_RUN_ABOVE, &allBitsCopy, &indexBits, &runBitsCopy);
    }
    else
    {
      dAveBitsPerPix[PALETTE_RUN_ABOVE] = std::numeric_limits<double>::max();
    }

    traIdx = m_pScanOrder[idx];    //unified position variable (raster scan)

    assert(RunValid || CopyValid);

    if( dAveBitsPerPix[PALETTE_RUN_ABOVE] <= dAveBitsPerPix[PALETTE_RUN_LEFT] )
    {
      pSPoint[traIdx] = PALETTE_RUN_ABOVE;
      pRun[traIdx]  = copyRun-1;

      pEscapeFlag[traIdx] = ( pcIndexBlock[traIdx] < 0 );

      Double error=0;
      m_paletteInfo[noElements].index    = 0;
      m_paletteInfo[noElements].position = idx;
      m_paletteInfo[noElements].paletteMode  = pSPoint[traIdx];
      m_paletteInfo[noElements].run      = pRun[traIdx];
      m_paletteInfo[noElements].bitsRun  = runBitsCopy;
      m_paletteInfo[noElements].bitsInd  = 0;
      m_paletteInfo[noElements].bitsAll  = allBitsCopy;

      if( pEscapeFlag[traIdx] )
      {
        xCalcPixelPred(pcCU, pOrg, paPixelValue, paRecoValue, width, strideOrg, traIdx);
      }
      idx++;

      temp = copyRun - 1;
      while (temp > 0)
      {
        traIdx = m_pScanOrder[idx];  //unified position variable (raster scan)

        pEscapeFlag[traIdx] = ( pcIndexBlock[traIdx] < 0 );

        if( pEscapeFlag[traIdx] )
        {
          xCalcPixelPred(pcCU, pOrg, paPixelValue, paRecoValue, width, strideOrg, traIdx);
        }

        pSPoint[traIdx] = PALETTE_RUN_ABOVE;
        idx++;

        temp--;
      }
      m_paletteInfo[noElements].error = error;
      noElements++;
    }
    else
    {
      pSPoint[traIdx] = PALETTE_RUN_LEFT;
      pRun[traIdx] = run;

      pEscapeFlag[traIdx] = ( pcIndexBlock[traIdx] < 0 );

      Double error=0;
      m_paletteInfo[noElements].position = idx;
      m_paletteInfo[noElements].paletteMode  = pSPoint[traIdx];
      m_paletteInfo[noElements].run      = pRun[traIdx];
      m_paletteInfo[noElements].bitsInd  = indexBits;
      m_paletteInfo[noElements].bitsRun  = runBitsIndex;
      m_paletteInfo[noElements].bitsAll  = allBitsIndex;
      m_paletteInfo[noElements].index    = pEscapeFlag[traIdx] ? (MAX_PALETTE_SIZE - 1) : pValue[traIdx];

      if (idx>0)
      {
        UInt uiTraIdxLeft = m_pScanOrder[idx - 1];
        if (pSPoint[uiTraIdxLeft] == PALETTE_RUN_LEFT)  ///< copy left
        {
          m_paletteInfo[noElements].indexPred = pValue[uiTraIdxLeft];
          if( pEscapeFlag[uiTraIdxLeft] )
          {
            m_paletteInfo[noElements].indexPred = indexMaxSize - 1;
          }
        }
        else
        {
          m_paletteInfo[noElements].indexPred = pValue[traIdx - width];
          if( pEscapeFlag[traIdx - width] )
          {
            m_paletteInfo[noElements].indexPred = indexMaxSize - 1;
          }
        }
      }

      if( pEscapeFlag[traIdx] )
      {
        xCalcPixelPred(pcCU, pOrg, paPixelValue, paRecoValue, width, strideOrg, traIdx);
      }

      idx++;

      temp = run;
      while (temp > 0)
      {
        traIdx = m_pScanOrder[idx];  //unified position variable (raster scan)

        pEscapeFlag[traIdx] = ( pcIndexBlock[traIdx] < 0 );

        if( pEscapeFlag[traIdx] )
        {
          xCalcPixelPred(pcCU, pOrg, paPixelValue, paRecoValue, width, strideOrg, traIdx);
        }

        pSPoint[traIdx] = PALETTE_RUN_LEFT;
        idx++;
        temp--;
      }
      m_paletteInfo[noElements].error=error;
      noElements++;
    }
  }
  m_paletteNoElements = noElements;
  assert (idx == total);
}

Double TEncSearch::xGetRunBits(TComDataCU* pcCU, Pel *pValue, UInt startPos, UInt run, PaletteRunMode paletteRunMode, UInt64 *allBits, UInt64 *indexBits, UInt64 *runBits)
{
  UInt width  = pcCU->getWidth(0);
  UInt height = pcCU->getHeight(0);
  UInt total = width * height;
  UInt curLevel = 0;
  UInt indexMaxSize = pcCU->getPaletteSize(COMPONENT_Y, 0);
  if( pcCU->getPaletteEscape(COMPONENT_Y, 0) )
  {
    indexMaxSize++;
  }

  UChar* pSPoint     = pcCU->getSPoint(COMPONENT_Y);

  UInt   traIdx    = m_pScanOrder[startPos];
  UInt   realLevel = pValue[traIdx];
  UChar* pEscapeFlag = pcCU->getEscapeFlag(COMPONENT_Y);
  m_pcRDGoOnSbacCoder->saveRestorePaletteCtx(0);
  m_pcEntropyCoder->resetBits();

  m_pcRDGoOnSbacCoder->encodeSPoint(startPos, width, pSPoint, m_pScanOrder);

  UInt64 sPointBits=m_pcRDGoOnSbacCoder->getNumPartialBits();

  assert(run >= 1);
  switch(paletteRunMode)
  {
  case PALETTE_RUN_LEFT:
    if( pEscapeFlag[traIdx] )
    {
      pValue[traIdx] = indexMaxSize - 1;
    }

    curLevel = m_pcRDGoOnSbacCoder->writePaletteIndex(startPos, pValue, indexMaxSize, pSPoint, width, pEscapeFlag);
    *indexBits = m_pcRDGoOnSbacCoder->getNumPartialBits() - sPointBits;
    if( pEscapeFlag[traIdx] )
    {
      pValue[traIdx] = realLevel;
    }
    m_pcRDGoOnSbacCoder->encodeRun((run - 1), PALETTE_RUN_LEFT, curLevel, total - startPos - 1);
    *runBits = m_pcRDGoOnSbacCoder->getNumPartialBits() - sPointBits - (*indexBits);

    break;
  case PALETTE_RUN_ABOVE:
    m_pcRDGoOnSbacCoder->encodeRun((run - 1), PALETTE_RUN_ABOVE, curLevel, total - startPos - 1);
    *runBits = m_pcRDGoOnSbacCoder->getNumPartialBits() - sPointBits;

    break;
  default:
    assert(0);
  }

  UInt bits = m_pcEntropyCoder->getNumberOfWrittenBits();
  *allBits=m_pcRDGoOnSbacCoder->getNumPartialBits();
  Double dCostPerPixel = bits * 1.0 / run;
  return dCostPerPixel;
}

Bool TEncSearch::isBlockVectorValid( Int xPos, Int yPos, Int width, Int height, TComDataCU *pcCU,
                                     Int xStartInCU, Int yStartInCU, Int xBv, Int yBv, Int ctuSize )
{
  static const Int s_floorLog2[65] =
  {
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3,
     3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
     4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
     4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 5, 5, 5, 6
  };

  Int ctuSizeLog2 = s_floorLog2[ctuSize];

  Int interpolationSamplesX = (pcCU->getPic()->getChromaFormat() == CHROMA_422 || pcCU->getPic()->getChromaFormat() == CHROMA_420) ?((xBv&0x1)<< 1) : 0;
  Int interpolationSamplesY = (pcCU->getPic()->getChromaFormat() == CHROMA_420) ? ((yBv&0x1)<< 1) : 0;
  Int refRightX  = xPos + xBv + width - 1 + interpolationSamplesX;
  Int refBottomY = yPos + yBv + height - 1 + interpolationSamplesY;
  Int picWidth = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
  Int picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();

  if ( (xPos + xBv - interpolationSamplesX) < 0 )
  {
    return false;
  }
  if ( refRightX >= picWidth )
  {
    return false;
  }
  if ( (yPos + yBv - interpolationSamplesY) < 0 )
  {
    return false;
  }
  if ( refBottomY >= picHeight )
  {
    return false;
  }

  if ( (xBv + width + interpolationSamplesX) > 0 && (yBv + height+interpolationSamplesY) > 0 )
  {
    return false;
  }

  // check same slice and tile
  TComPicSym *pcSym = pcCU->getPic()->getPicSym();
  TComDataCU* tlCtu = pcSym->getCtu(((xPos + xBv - interpolationSamplesX) >> ctuSizeLog2) + ((yPos + yBv - interpolationSamplesY) >> ctuSizeLog2) * pcSym->getFrameWidthInCtus());
  TComDataCU* rbCtu = pcSym->getCtu((refRightX >> ctuSizeLog2) + (refBottomY >> ctuSizeLog2) * pcSym->getFrameWidthInCtus());
  if (!pcCU->CUIsFromSameSliceAndTile(tlCtu) || !pcCU->CUIsFromSameSliceAndTile(rbCtu))
  {
    return false;
  }

  if ( refBottomY>>ctuSizeLog2 < yPos>>ctuSizeLog2 )
  {
    Int uiRefCuX   = refRightX/ctuSize;
    Int uiRefCuY   = refBottomY/ctuSize;
    Int uiCuPelX   = xPos / ctuSize;
    Int uiCuPelY   = yPos / ctuSize;

    if(((Int)(uiRefCuX - uiCuPelX) > (Int)((uiCuPelY - uiRefCuY))))
    {
      return false;
    }
    else
    {
      return true;
    }
  }

  if ( refBottomY>>ctuSizeLog2 > yPos>>ctuSizeLog2 )
  {
    return false;
  }

  // in the same CTU line
  if ( refRightX>>ctuSizeLog2 < xPos>>ctuSizeLog2 )
  {
    return true;
  }
  if ( refRightX>>ctuSizeLog2 > xPos>>ctuSizeLog2 )
  {
    return false;
  }

  // same CTU
  Int mask = 1<<ctuSizeLog2;
  mask -= 1;
  Int rasterCurr = ( ( ((yPos&mask) - yStartInCU)>>2 ) << (ctuSizeLog2-2) ) + ( ((xPos&mask) - xStartInCU)>>2 );
  Int rasterRef  = ( ( (refBottomY&mask)>>2 ) << (ctuSizeLog2-2) ) + ( (refRightX&mask)>>2 );

  if ( g_auiRasterToZscan[rasterRef] >= g_auiRasterToZscan[rasterCurr] )
  {
    return false;
  }
  return true;
}

// based on predInterSearch()
Bool TEncSearch::predIntraBCSearch( TComDataCU * pcCU,
                                    TComYuv    * pcOrgYuv,
                                    TComYuv    *&rpcPredYuv,
                                    TComYuv    *&rpcResiYuv,
                                    TComYuv    *&rpcRecoYuv
                                    DEBUG_STRING_FN_DECLARE(sDebug),
                                    Bool         bUse1DSearchFor8x8,
                                    Bool         bUseRes,
                                    Bool         testOnlyPred
                                    )
{
  rpcPredYuv->clear();
  if ( !bUseRes )
  {
    rpcResiYuv->clear();
  }
  rpcRecoYuv->clear();

  PartSize     ePartSize  = pcCU->getPartitionSize( 0 );

  if ( m_pcEncCfg->getUseIntraBlockCopyFastSearch() && (pcCU->getWidth( 0 ) > SCM_S0067_MAX_CAND_SIZE) )
  {
    return false;
  }

  const Int numPart = pcCU->getNumPartitions();
  Distortion totalCost = 0;
  for( Int partIdx = 0; partIdx < numPart; ++partIdx )
  {
    Int width, height;
    UInt partAddr = 0;
    pcCU->getPartIndexAndSize( partIdx, partAddr, width, height );

    TComMvField cMEMvField;
    Distortion  cost;

    TComMv      cMv, cMvd, cMvPred[2];
    AMVPInfo currAMVPInfo;
    pcCU->fillMvpCand( partIdx, partAddr, REF_PIC_LIST_0, pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, &currAMVPInfo );
    cMvPred[0].set( currAMVPInfo.m_acMvCand[0].getHor() >> 2, currAMVPInfo.m_acMvCand[0].getVer() >> 2);
    cMvPred[1].set( currAMVPInfo.m_acMvCand[1].getHor() >> 2, currAMVPInfo.m_acMvCand[1].getVer() >> 2);

    xIntraBlockCopyEstimation ( pcCU, pcOrgYuv, partIdx, cMvPred, cMv, cost, bUse1DSearchFor8x8, testOnlyPred );

    if( m_pcEncCfg->getUseHashBasedIntraBCSearch()
      && pcCU->getWidth(0) == 8
      && !testOnlyPred
      && ePartSize == SIZE_2Nx2N
      && (m_pcEncCfg->getUseHashBasedIntraBCSearch()
        || pcCU->getIntraBCSearchAreaWidth( m_pcEncCfg->getIntraBCSearchWidthInCTUs() ) != pcCU->getIntraBCSearchAreaWidth( m_pcEncCfg->getIntraBCNonHashSearchWidthInCTUs() ))
      )
    {
      Distortion intraBCECost = cost;
      xIntraBCHashSearch ( pcCU, pcOrgYuv, partIdx, cMvPred, cMv, (UInt)intraBCECost);
      cost = std::min(intraBCECost, cost);
    }
    totalCost += cost;
    // choose one MVP and compare with merge mode
    // no valid intra BV
    if ( cMv.getHor() == 0 && cMv.getVer() == 0 )
    {
      if (testOnlyPred)
      {
        m_lastCandCost = MAX_UINT;
      }
      return false;
    }

    m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( 0 ) );
    m_pcRdCost->setCostScale( 0 );

    UInt depth = pcCU->getDepth( 0 );
    Int bitsAMVPBest, bitsAMVPTemp,                bitsMergeTemp;
    Int distAMVPBest,                              distMergeTemp;
    Int costAMVPBest,               costMergeBest, costMergeTemp;
    bitsAMVPBest = MAX_INT;
    costAMVPBest = MAX_INT;
    costMergeBest = MAX_INT;
    Int mvpIdxBest = 0;
    Int mvpIdxTemp;
    Int mrgIdxBest = -1;
    Int mrgIdxTemp = -1;
    Int xCUStart = pcCU->getCUPelX();
    Int yCUStart = pcCU->getCUPelY();
    Int xStartInCU, yStartInCU;
    pcCU->getStartPosition( partIdx, xStartInCU, yStartInCU );
    Pel* pCurrStart;
    Pel* pCurr;
    Pel* pRef;
    Int currStride, refStride;
    distAMVPBest = 0;

    TComMv cMvQuaterPixl = cMv;
    cMvQuaterPixl <<= 2;
    pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMvQuaterPixl, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( TComMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
    pcCU->setInterDirSubParts( 1 + REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth(0) );
    motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );

    for (UInt ch = COMPONENT_Y; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
    {
      Int tempHeight = height >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));
      Int tempWidth = width >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));

      pCurrStart = pcOrgYuv->getAddr  ( ComponentID(ch), partAddr );
      currStride = pcOrgYuv->getStride( ComponentID(ch) );
      pCurr = pCurrStart;
      ComponentID compID = (ComponentID)ch;
      pRef = m_tmpYuvPred.getAddr( compID, partAddr);
      refStride = m_tmpYuvPred.getStride(compID);
      distAMVPBest += getSAD( pRef, refStride, pCurr, currStride, tempWidth, tempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );
    }

    cMvPred[0].setHor( currAMVPInfo.m_acMvCand[0].getHor() >> 2);
    cMvPred[0].setVer( currAMVPInfo.m_acMvCand[0].getVer() >> 2);
    cMvPred[1].setHor( currAMVPInfo.m_acMvCand[1].getHor() >> 2);
    cMvPred[1].setVer( currAMVPInfo.m_acMvCand[1].getVer() >> 2);

    for ( mvpIdxTemp=0; mvpIdxTemp<currAMVPInfo.iN; mvpIdxTemp++ )
    {
      m_pcRdCost->setPredictor( cMvPred[mvpIdxTemp] );
      bitsAMVPTemp = m_pcRdCost->getBitsOfVectorWithPredictor( cMv.getHor(), cMv.getVer() );
      if ( bitsAMVPTemp < bitsAMVPBest )
      {
        bitsAMVPBest = bitsAMVPTemp;
        mvpIdxBest = mvpIdxTemp;
      }
    }
    bitsAMVPBest++; // for MVP Index bits
    costAMVPBest = distAMVPBest + m_pcRdCost->getCost( bitsAMVPBest );

    TComMvField cMvFieldNeighbours[MRG_MAX_NUM_CANDS << 1]; // double length for mv of both lists
    UChar uhInterDirNeighbours[MRG_MAX_NUM_CANDS];
    Int numValidMergeCand = 0;

    if ( ePartSize != SIZE_2Nx2N )
    {
      if ( pcCU->getSlice()->getPPS()->getLog2ParallelMergeLevelMinus2() && ePartSize != SIZE_2Nx2N && pcCU->getWidth( 0 ) <= 8 )
      {
        pcCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth );
        if ( partIdx == 0 )
        {
#if MCTS_ENC_CHECK
          UInt numSpatialMergeCandidates = 0;
          pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
          if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
          {
            numValidMergeCand = numSpatialMergeCandidates;
          }
#else
          pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );
#endif
        }
        pcCU->setPartSizeSubParts( ePartSize, 0, depth );
      }
      else
      {
#if MCTS_ENC_CHECK
        UInt numSpatialMergeCandidates = 0;
        pcCU->getInterMergeCandidates( partAddr, partIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
        if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
        {
          numValidMergeCand = numSpatialMergeCandidates;
        }
#else
        pcCU->getInterMergeCandidates( partAddr, partIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );
#endif
      }

      pcCU->roundMergeCandidates(cMvFieldNeighbours, numValidMergeCand);
      xRestrictBipredMergeCand( pcCU, partIdx, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );

      for ( mrgIdxTemp = 0; mrgIdxTemp < numValidMergeCand; mrgIdxTemp++ )
      {
        if ( uhInterDirNeighbours[mrgIdxTemp] != 1 )
        {
          continue;
        }

        if ( pcCU->getSlice()->getRefPic( REF_PIC_LIST_0, cMvFieldNeighbours[mrgIdxTemp<<1].getRefIdx() )->getPOC() != pcCU->getSlice()->getPOC() )
        {
          continue;
        }

        if ( !isBlockVectorValid( xCUStart+xStartInCU, yCUStart+yStartInCU, width, height, pcCU,
          xStartInCU, yStartInCU, (cMvFieldNeighbours[mrgIdxTemp<<1].getHor() >> 2), (cMvFieldNeighbours[mrgIdxTemp<<1].getVer()>>2), pcCU->getSlice()->getSPS()->getMaxCUWidth() ) )
        {
          continue;
        }
        bitsMergeTemp = mrgIdxTemp == m_pcEncCfg->getMaxNumMergeCand() ? mrgIdxTemp : mrgIdxTemp+1;

        distMergeTemp = 0;

        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMvFieldNeighbours[mrgIdxTemp<<1].getMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( TComMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->setInterDirSubParts( 1 + REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth(0) );
        motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );

        for (UInt ch = COMPONENT_Y; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
        {
          Int tempHeight = height >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));
          Int tempWidth = width >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));

          pCurrStart = pcOrgYuv->getAddr  ( ComponentID(ch), partAddr );
          currStride = pcOrgYuv->getStride( ComponentID(ch) );
          pCurr = pCurrStart;

          ComponentID compID = (ComponentID)ch;
          pRef = m_tmpYuvPred.getAddr( compID, partAddr);
          refStride = m_tmpYuvPred.getStride(compID);
          distMergeTemp += getSAD( pRef, refStride, pCurr, currStride, tempWidth, tempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );
        }
        costMergeTemp = distMergeTemp + m_pcRdCost->getCost( bitsMergeTemp );

        if ( costMergeTemp < costMergeBest )
        {
          costMergeBest = costMergeTemp;
          mrgIdxBest = mrgIdxTemp;
        }
      }
    }

    if ( costAMVPBest < costMergeBest )
    {
      TComMv zeroMv;
      TComMv mv( (cMv.getHor()<<2), (cMv.getVer()<<2) );
      TComMvField mvField[2];
      mvField[0].setMvField( mv, pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1 );   // the current picture is at the last position of list0
      mvField[1].setMvField( zeroMv, -1 );

      pcCU->setMergeFlagSubParts( false, partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( mvField[0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( mvField[1], ePartSize, partAddr, 0, partIdx );

      TComMv mvd( cMv.getHor() - (currAMVPInfo.m_acMvCand[mvpIdxBest].getHor() >> 2), cMv.getVer() - (currAMVPInfo.m_acMvCand[mvpIdxBest].getVer()>>2) );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( mvd,    ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( zeroMv, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( mvpIdxBest, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
    }
    else
    {
      TComMv zeroMv;
      TComMv mv( cMvFieldNeighbours[mrgIdxBest<<1].getHor(), cMvFieldNeighbours[mrgIdxBest<<1].getVer() );
      TComMvField mvField[2];
      mvField[0].setMvField( mv, pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1 );   // the current picture is at the last position of list0
      mvField[1].setMvField( zeroMv, -1 );

      pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );
      pcCU->setMergeIndexSubParts( mrgIdxBest, partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( mvField[0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( mvField[1], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( zeroMv, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( zeroMv, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
    }
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif
  }

  Distortion abortThreshold = pcCU->getWidth(0)*pcCU->getHeight(0)*2;
  if( testOnlyPred )
  {
    if( numPart==1 && totalCost > abortThreshold )
    {
      m_lastCandCost = MAX_UINT;
      return false;
    }
    m_lastCandCost = totalCost;
  }
  else if( totalCost < abortThreshold && 3*totalCost>>2 >= m_lastCandCost )
  {
    return false;
  }

  // motion compensation
  if( !pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() || !m_pcEncCfg->getRGBFormatFlag() || (pcCU->getCUTransquantBypass(0) && (pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) != pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_CHROMA))) )
  {
    motionCompensation ( pcCU, rpcPredYuv );
  }
  return true;
}

// based on predInterSearch()
Bool TEncSearch::predMixedIntraBCInterSearch( TComDataCU * pcCU,
                                              TComYuv    * pcOrgYuv,
                                              TComYuv    *&rpcPredYuv,
                                              TComYuv    *&rpcResiYuv,
                                              TComYuv    *&rpcRecoYuv
                                              DEBUG_STRING_FN_DECLARE( sDebug ),
                                              TComMv*      iMvCandList,
                                              Bool         bUseRes
                                              )
{
  rpcPredYuv->clear();
  if ( !bUseRes )
  {
    rpcResiYuv->clear();
  }
  rpcRecoYuv->clear();

  PartSize     ePartSize  = pcCU->getPartitionSize( 0 );

  UInt depth = pcCU->getDepth( 0 );
  Int numComb = 2;
  Int numPart = 2;
  //  PredMode pMixedMode[2][2] = {{MODE_INTRABC, MODE_INTER},{MODE_INTER, MODE_INTRABC}};
  Distortion  cost[2]={ 0, 0 };
  Distortion maxCost = std::numeric_limits<Distortion>::max();

  TComPattern   cPattern;

  TComMvField cMvFieldNeighbours[2][MRG_MAX_NUM_CANDS << 1]; // double length for mv of both lists
  UChar uhInterDirNeighbours[2][MRG_MAX_NUM_CANDS];
  Int numValidMergeCand[2] ={ MRG_MAX_NUM_CANDS, MRG_MAX_NUM_CANDS };
  Int numPredDir = pcCU->getSlice()->isInterP() ? 1 : 2;
  TComMv       cMvZero( 0, 0 );

  TComMv  cMvPredCand[2][2];
  Int IBCValidFlag = 0;
  Int bestIBCMvpIdx[2] ={ 0, 0 };
  Int bestInterMvpIdx[2] ={ 0, 0 };
  Int bestInterDir[2] ={ 0, 0 };
  Int bestRefIdx[2] ={ 0, 0 };
  Bool isMergeMode[2]={ false, false };
  Bool isIBCMergeMode[2]={ false, false };
  TComMvField cMRGMvField[2][2];
  TComMvField cMRGMvFieldIBC[2][2];

  for ( Int combo = 0; combo < numComb; combo++ ) // number of combination
  {
    for ( Int partIdx = 0; partIdx < numPart; ++partIdx )
    {
      Int dummyWidth, dummyHeight;
      UInt partAddr = 0;
      pcCU->getPartIndexAndSize( partIdx, partAddr, dummyWidth, dummyHeight );

      //  Search key pattern initialization
#if MCTS_ENC_CHECK
      Int roiPosX, roiPosY;
      Int roiW, roiH;
      pcCU->getPartPosition(partIdx, roiPosX, roiPosY, roiW, roiH);
      assert(roiW == dummyWidth);
      assert(roiH == dummyHeight);
      cPattern.initPattern( pcOrgYuv->getAddr(COMPONENT_Y, partAddr),
                            dummyWidth,
                            dummyHeight,
                            pcOrgYuv->getStride(COMPONENT_Y),
                            pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA),
                            roiPosX,
                            roiPosY);
      xInitTileBorders(pcCU, &cPattern);
#else
      cPattern.initPattern( pcOrgYuv->getAddr( COMPONENT_Y, partAddr ), dummyWidth, dummyHeight, pcOrgYuv->getStride( COMPONENT_Y ), pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
#endif

      // disable weighted prediction
      setWpScalingDistParam( pcCU, -1, REF_PIC_LIST_X );


      TComMv mvPred[2]; // put result of AMVP into mvPred
      TComMv bvPred[2]; // put result of BVP into bvPred
      Distortion   biPDistTemp = std::numeric_limits<Distortion>::max();
      if ( (combo==0 && partIdx==0) || (combo==1 && partIdx==1) ) // intraBC
      {
        TComMv cMv = iMvCandList[8+partIdx];
        if ( cMv.getHor() == 0 && cMv.getVer() == 0 )
        {
          cost[combo] = maxCost;
          IBCValidFlag++;
          break;
        }
        m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( partAddr ) );
        m_pcRdCost->setCostScale( 0 );
        AMVPInfo currAMVPInfo;
        xEstimateMvPredAMVP( pcCU, pcOrgYuv, partIdx, REF_PIC_LIST_0, pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, bvPred[0], false, &biPDistTemp );

        bvPred[0] = pcCU->getCUMvField( REF_PIC_LIST_0 )->getAMVPInfo()->m_acMvCand[0];
        bvPred[1] = pcCU->getCUMvField( REF_PIC_LIST_0 )->getAMVPInfo()->m_acMvCand[1];
        bvPred[0] >>= 2;
        bvPred[1] >>= 2;

        /////////////////////////////////////////////////////////////
        // ibc merge
        // choose one MVP and compare with merge mode

        m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( 0 ) );
        m_pcRdCost->setCostScale( 0 );


        //  UInt depth = pcCU->getDepth( 0 );
        Int bitsAMVPBest, bitsAMVPTemp, bitsMergeTemp;
        Int distAMVPBest, distMergeTemp;
        Int costAMVPBest, costMergeBest, costMergeTemp;
        bitsAMVPBest = MAX_INT;
        costAMVPBest = MAX_INT;
        costMergeBest = MAX_INT;
        Int mvpIdxBest = 0;
        Int mvpIdxTemp;
        Int mrgIdxBest = -1;
        Int mrgIdxTemp = -1;
        Int xCUStart = pcCU->getCUPelX();
        Int yCUStart = pcCU->getCUPelY();
        Int xStartInCU, yStartInCU;
        pcCU->getStartPosition( partIdx, xStartInCU, yStartInCU );
        Pel* pCurrStart;
        Int currStride;
        Pel* pCurr;
        Int refStride;
        Pel* pRef;
        distAMVPBest = 0;

        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMv, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( TComMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
        pcCU->setInterDirSubParts( 1 + REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth(0) );
        motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );

        for (UInt ch = COMPONENT_Y; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
        {
          Int tempWidth = dummyWidth >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));
          Int tempHeight = dummyHeight >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));

          pCurrStart = pcOrgYuv->getAddr( ComponentID(ch), partAddr);
          currStride = pcOrgYuv->getStride( ComponentID(ch) );
          pCurr = pCurrStart;

          ComponentID compID = (ComponentID)ch;
          pRef = m_tmpYuvPred.getAddr( compID, partAddr);
          refStride = m_tmpYuvPred.getStride(compID);
          distAMVPBest += getSAD( pRef, refStride, pCurr, currStride, tempWidth, tempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );
        }

        for ( mvpIdxTemp=0; mvpIdxTemp<AMVP_MAX_NUM_CANDS; mvpIdxTemp++ )
        {
          m_pcRdCost->setPredictor( bvPred[mvpIdxTemp] );
          bitsAMVPTemp = m_pcRdCost->getBitsOfVectorWithPredictor( (cMv.getHor() >> 2), (cMv.getVer() >> 2) );
          if ( bitsAMVPTemp < bitsAMVPBest )
          {
            bitsAMVPBest = bitsAMVPTemp;
            mvpIdxBest = mvpIdxTemp;
          }
        }

        bitsAMVPBest++; // for MVP Index bits
        costAMVPBest = distAMVPBest + m_pcRdCost->getCost( bitsAMVPBest );

        TComMvField cMvFieldNeighboursIBC[MRG_MAX_NUM_CANDS << 1]; // double length for mv of both lists
        UChar uhInterDirNeighboursIBC[MRG_MAX_NUM_CANDS];
        Int numValidMergeCandIBC = 0;

        if ( ePartSize != SIZE_2Nx2N )
        {
          if ( pcCU->getSlice()->getPPS()->getLog2ParallelMergeLevelMinus2() && ePartSize != SIZE_2Nx2N && pcCU->getWidth( 0 ) <= 8 )
          {
            pcCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth );
            if ( partIdx == 0 )
            {
#if MCTS_ENC_CHECK
              UInt numSpatialMergeCandidates = 0;
              pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighboursIBC, uhInterDirNeighboursIBC, numValidMergeCandIBC, numSpatialMergeCandidates );
              if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
              {
                numValidMergeCandIBC = numSpatialMergeCandidates;
              }
#else
              pcCU->getInterMergeCandidates( 0, 0, cMvFieldNeighboursIBC, uhInterDirNeighboursIBC, numValidMergeCandIBC );
#endif
            }
            pcCU->setPartSizeSubParts( ePartSize, 0, depth );
          }
          else
          {
#if MCTS_ENC_CHECK
            UInt numSpatialMergeCandidates = 0;
            pcCU->getInterMergeCandidates( partAddr, partIdx, cMvFieldNeighboursIBC, uhInterDirNeighboursIBC, numValidMergeCandIBC, numSpatialMergeCandidates );
            if (m_pcEncCfg->getTMCTSSEITileConstraint() && pcCU->isLastColumnCTUInTile())
            {
              numValidMergeCandIBC = numSpatialMergeCandidates;
            }
#else
            pcCU->getInterMergeCandidates( partAddr, partIdx, cMvFieldNeighboursIBC, uhInterDirNeighboursIBC, numValidMergeCandIBC );
#endif
          }

          pcCU->roundMergeCandidates(cMvFieldNeighboursIBC, numValidMergeCandIBC);
          xRestrictBipredMergeCand( pcCU, partIdx, cMvFieldNeighboursIBC, uhInterDirNeighboursIBC, numValidMergeCandIBC );

          for ( mrgIdxTemp = 0; mrgIdxTemp < numValidMergeCandIBC; mrgIdxTemp++ )
          {
            if ( uhInterDirNeighboursIBC[mrgIdxTemp] != 1 )
            {
              continue;
            }

            if ( pcCU->getSlice()->getRefPic( REF_PIC_LIST_0, cMvFieldNeighboursIBC[mrgIdxTemp<<1].getRefIdx() )->getPOC() != pcCU->getSlice()->getPOC() )
            {
              continue;
            }

            if ( !isBlockVectorValid( xCUStart+xStartInCU, yCUStart+yStartInCU, dummyWidth, dummyHeight, pcCU,
              xStartInCU, yStartInCU, (cMvFieldNeighboursIBC[mrgIdxTemp<<1].getHor() >> 2), (cMvFieldNeighboursIBC[mrgIdxTemp<<1].getVer() >> 2), pcCU->getSlice()->getSPS()->getMaxCUWidth() ) )
            {
              continue;
            }

            bitsMergeTemp = mrgIdxTemp == m_pcEncCfg->getMaxNumMergeCand() ? mrgIdxTemp : mrgIdxTemp+1;

            distMergeTemp = 0;
            pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMvFieldNeighboursIBC[mrgIdxTemp<<1].getMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
            pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
            pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( TComMv(), pcCU->getPartitionSize(0), partAddr, 0, partIdx );
            pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, pcCU->getPartitionSize(0), partAddr, 0, partIdx );
            pcCU->setInterDirSubParts( 1 + REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth(0) );
            motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );

            for (UInt ch = COMPONENT_Y; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
            {
              Int tempWidth = dummyWidth >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));
              Int tempHeight = dummyHeight >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));

              pCurrStart = pcOrgYuv->getAddr( ComponentID(ch), partAddr);
              currStride = pcOrgYuv->getStride( ComponentID(ch) );
              pCurr = pCurrStart;
              ComponentID compID = (ComponentID)ch;
              pRef = m_tmpYuvPred.getAddr( compID, partAddr);
              refStride = m_tmpYuvPred.getStride(compID);
              distMergeTemp += getSAD( pRef, refStride, pCurr, currStride, tempWidth , tempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );
            }

            costMergeTemp = distMergeTemp + m_pcRdCost->getCost( bitsMergeTemp );

            if ( costMergeTemp < costMergeBest )
            {
              costMergeBest = costMergeTemp;
              mrgIdxBest = mrgIdxTemp;
            }
          }
        }

        if ( costMergeBest < costAMVPBest )
        {
          cost[combo] += costMergeBest;
          isIBCMergeMode[combo] = true;
          bestIBCMvpIdx[combo] = mrgIdxBest;

          TComMvField mvField[2];
          TComMv mv( cMvFieldNeighboursIBC[mrgIdxBest<<1].getHor(), cMvFieldNeighboursIBC[mrgIdxBest<<1].getVer() );
          mvField[0].setMvField( mv, pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1 );   // the current picture is at the last position of list0
          mvField[1].setMvField( cMvZero, -1 );
          cMRGMvFieldIBC[combo][0] = mvField[0];
          cMRGMvFieldIBC[combo][1] = mvField[1];
        }
        else
        {
          cost[combo] += costAMVPBest;
          isIBCMergeMode[combo] = false;
          bestIBCMvpIdx[combo] = mvpIdxBest;
          cMvPredCand[combo][partIdx].setHor( bvPred[mvpIdxBest].getHor() << 2);
          cMvPredCand[combo][partIdx].setVer( bvPred[mvpIdxBest].getVer() << 2);
        }

        pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
        if ( isIBCMergeMode[combo] )
        {
          pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMRGMvFieldIBC[combo][0].getMv(), ePartSize, partAddr, 0, partIdx );
        }
        else
        {
          pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( iMvCandList[8+partIdx], ePartSize, partAddr, 0, partIdx );
          pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, ePartSize, partAddr, 0, partIdx );
          pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
        }
        // ibc merge
        /////////////////////////////////////////////////////////////
      }
      else // is inter PU
      {
        Distortion  costInterTemp = 0;
        Distortion  costInterBest = std::numeric_limits<Distortion>::max();
        m_pcRdCost->setCostScale( 0 );
        //  Uni-directional prediction
        //  Int numPredDir = 1;
        for ( Int refList = 0; refList < numPredDir; refList++ )
        {
          RefPicList  eRefPicList = (refList ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
          //  UInt numRef = (pcCU->getSlice()->getNumRefIdx(eRefPicList)-1 > 1) ? 2: 1;
          UInt numRef = refList ? ((pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_1 ) > 1) ? 2 : 1) : ((pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 )-1 > 1) ? 2 : 1);
          for ( Int refIdx = 0; refIdx < numRef; refIdx++ )
          {
            TComMv cMv = iMvCandList[4*refList + 2*refIdx + partIdx];

            xEstimateMvPredAMVP( pcCU, pcOrgYuv, partIdx, eRefPicList, refIdx, mvPred[0], false, &biPDistTemp );
            Int mvpIdx;// = pcCU->getMVPIdx(eRefPicList, partAddr);

            Distortion  tempCost0 = 0;
            Distortion  tempCost1 = 0;
            mvPred[0] = pcCU->getCUMvField( eRefPicList )->getAMVPInfo()->m_acMvCand[0];
            mvPred[1] = pcCU->getCUMvField( eRefPicList )->getAMVPInfo()->m_acMvCand[1];

            m_pcRdCost->setPredictor( mvPred[0] );
            tempCost0 = m_pcRdCost->getCostOfVectorWithPredictor( cMv.getHor(), cMv.getVer() );
            m_pcRdCost->setPredictor( mvPred[1] );
            tempCost1 = m_pcRdCost->getCostOfVectorWithPredictor( cMv.getHor(), cMv.getVer() );
            if ( tempCost1 < tempCost0 )
            {
              mvpIdx = 1;
            }
            else
            {
              mvpIdx = 0;
            }

            UInt bitsTemp = m_auiMVPIdxCost[mvpIdx][AMVP_MAX_NUM_CANDS];

            m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( partAddr ) );
            m_pcRdCost->setPredictor( mvPred[mvpIdx] );
            if( pcCU->getSlice()->getUseIntegerMv() )
            {
              pcCU->getCUMvField( eRefPicList )->setAllMv( (cMv>>2)<<2, ePartSize, partAddr, 0, partIdx );
            }
            else
            {
              pcCU->getCUMvField( eRefPicList )->setAllMv( cMv, ePartSize, partAddr, 0, partIdx );
            }
            pcCU->getCUMvField( eRefPicList )->setAllRefIdx( refIdx, ePartSize, partAddr, 0, partIdx );
            pcCU->setInterDirSubParts( 1 + refList, partAddr, partIdx, depth );

            motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );
            costInterTemp = 0;
            for (Int ch = COMPONENT_Y; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
            {
              Int tempWidth = dummyWidth >> pcCU->getPic()->getPicYuvRec()->getComponentScaleX(ComponentID(ch));
              Int tempHeight = dummyHeight >> pcCU->getPic()->getPicYuvRec()->getComponentScaleY(ComponentID(ch));

              Int refStride = m_tmpYuvPred.getStride(ComponentID(ch));
              Int orgStride = pcOrgYuv->getStride(ComponentID(ch));
              Pel *pRef =  m_tmpYuvPred.getAddr( ComponentID(ch), partAddr);
              Pel *pOrg = pcOrgYuv->getAddr( ComponentID(ch), partAddr);
              //calculate the dist;
              costInterTemp += getSAD( pRef, refStride, pOrg, orgStride, tempWidth, tempHeight, pcCU->getSlice()->getSPS()->getBitDepths() );

              if (costInterTemp >= costInterBest)
              {
                break;
              }
            }
            pcCU->getCUMvField( eRefPicList )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
            //  m_pcRdCost->setCostScale  ( 2 );
            costInterTemp += m_pcRdCost->getCostOfVectorWithPredictor( cMv.getHor(), cMv.getVer() );
            costInterTemp += m_pcRdCost->getCost( bitsTemp );

            if ( costInterTemp < costInterBest )
            {
              costInterBest = costInterTemp;
              bestInterMvpIdx[combo] = mvpIdx;
              bestInterDir[combo] = refList;
              bestRefIdx[combo] = refIdx;
              cMvPredCand[combo][partIdx] = mvPred[mvpIdx];
            }
          }
        } // end RefIdx and RefList search

        UInt MRGInterDir = 0;
        UInt MRGIndex = 0;

        // find Merge result
        Distortion MRGCost = std::numeric_limits<Distortion>::max();
        pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );

        xMergeEstimation( pcCU, pcOrgYuv, partIdx, MRGInterDir, cMRGMvField[combo], MRGIndex, MRGCost, cMvFieldNeighbours[combo], uhInterDirNeighbours[combo], numValidMergeCand[combo], 1 );
        pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
        pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );

        if ( MRGCost < costInterBest )
        {
          costInterBest = MRGCost;
          isMergeMode[combo] = true;
          bestInterMvpIdx[combo] = MRGIndex;
          bestInterDir[combo]= MRGInterDir;
        }

        cost[combo] += costInterBest;
        if ( isMergeMode[combo] )
        {
          pcCU->setInterDirSubParts( bestInterDir[combo], partAddr, partIdx, depth );
          pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvField[combo][0], ePartSize, partAddr, 0, partIdx );
          pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvField[combo][1], ePartSize, partAddr, 0, partIdx );
        }
        else
        {
          Int refListOpt = bestInterDir[combo];
          Int refIdxOpt = bestRefIdx[combo];
          RefPicList  eRefPicListOpt = (refListOpt ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
          if( pcCU->getSlice()->getUseIntegerMv() )
          {
            pcCU->getCUMvField( eRefPicListOpt )->setAllMv( (iMvCandList[partIdx + 2*refIdxOpt + 4*refListOpt]>>2)<<2, ePartSize, partAddr, 0, partIdx );
          }
          else
          {
            pcCU->getCUMvField( eRefPicListOpt )->setAllMv( iMvCandList[partIdx + 2*refIdxOpt + 4*refListOpt], ePartSize, partAddr, 0, partIdx );
          }
          pcCU->getCUMvField( eRefPicListOpt )->setAllRefIdx( refIdxOpt, ePartSize, partAddr, 0, partIdx );
          pcCU->getCUMvField( (RefPicList)(1-refListOpt) )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
          pcCU->setInterDirSubParts( 1 + refListOpt, partAddr, partIdx, depth );
          pcCU->setMVPIdxSubParts( bestInterMvpIdx[combo], eRefPicListOpt, partAddr, partIdx, depth );
        }
      }
    } // for ipartIdx
  } // for combo

  if ( IBCValidFlag > 1 )
  {
    return false;
  }

  TComMv cMvd;
  TComMv cMVFinal;
  if ( cost[0] <= cost[1] )
  {
    Int iDummyWidth1, iDummyHeight1;
    UInt partAddr = 0;
    UInt partIdx = 0;
    pcCU->getPartIndexAndSize( partIdx, partAddr, iDummyWidth1, iDummyHeight1 );

    if ( isIBCMergeMode[0] )
    {
      pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );
      pcCU->setMergeIndexSubParts( bestIBCMvpIdx[0], partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvFieldIBC[0][0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvFieldIBC[0][1], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
    }
    else
    {
      pcCU->setMergeFlagSubParts( false, partAddr, partIdx, depth );
      cMvd.setHor( (iMvCandList[8].getHor() - cMvPredCand[0][0].getHor()) >> 2 );
      cMvd.setVer( (iMvCandList[8].getVer() - cMvPredCand[0][0].getVer()) >> 2 );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( iMvCandList[8], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvd, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( bestIBCMvpIdx[0], REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
    }
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif

    partIdx = 1;
    pcCU->getPartIndexAndSize( partIdx, partAddr, iDummyWidth1, iDummyHeight1 );

    if ( isMergeMode[0] )
    {
      pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );
      pcCU->setMergeIndexSubParts( bestInterMvpIdx[0], partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( bestInterDir[0], partAddr, partIdx, depth );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvField[0][0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvField[0][1], ePartSize, partAddr, 0, partIdx );

      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );

      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
      pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
    }
    else
    {
      Int refListOpt = bestInterDir[0];
      Int refIdxOpt = bestRefIdx[0];
      RefPicList  eRefPicListOpt = (refListOpt ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        cMvd.setHor( (iMvCandList[1 + 2*refIdxOpt + 4*refListOpt].getHor() >> 2) - (cMvPredCand[0][1].getHor()>>2) );
        cMvd.setVer( (iMvCandList[1 + 2*refIdxOpt + 4*refListOpt].getVer() >> 2) - (cMvPredCand[0][1].getVer()>>2) );
        pcCU->getCUMvField( eRefPicListOpt )->setAllMv( (iMvCandList[1 + 2*refIdxOpt + 4*refListOpt]>>2)<<2, ePartSize, partAddr, 0, partIdx );
      }
      else
      {
        cMvd.setHor( iMvCandList[1 + 2*refIdxOpt + 4*refListOpt].getHor() - cMvPredCand[0][1].getHor() );
        cMvd.setVer( iMvCandList[1 + 2*refIdxOpt + 4*refListOpt].getVer() - cMvPredCand[0][1].getVer() );
        pcCU->getCUMvField( eRefPicListOpt )->setAllMv( iMvCandList[1 + 2*refIdxOpt + 4*refListOpt], ePartSize, partAddr, 0, partIdx );
      }
      pcCU->getCUMvField( eRefPicListOpt )->setAllMvd( cMvd, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( eRefPicListOpt )->setAllRefIdx( refIdxOpt, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( (RefPicList)(1-refListOpt) )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
      pcCU->setInterDirSubParts( 1 + refListOpt, partAddr, partIdx, depth );
      pcCU->setMergeFlagSubParts( false, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( bestInterMvpIdx[0], eRefPicListOpt, partAddr, partIdx, depth );
    }
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif
  }
  else
  {
    Int dummyWidth2, dummyHeight2;
    UInt partAddr = 0;
    UInt partIdx = 0;

    pcCU->getPartIndexAndSize( partIdx, partAddr, dummyWidth2, dummyHeight2 );

    if ( isMergeMode[1] )
    {
      pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );
      pcCU->setMergeIndexSubParts( bestInterMvpIdx[1], partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( bestInterDir[1], partAddr, partIdx, depth );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvField[1][0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvField[1][1], ePartSize, partAddr, 0, partIdx );

      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );

      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
      pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );
    }
    else
    {
      Int refListOpt = bestInterDir[1];
      Int refIdxOpt = bestRefIdx[1];
      RefPicList  eRefPicListOpt = (refListOpt ? REF_PIC_LIST_1 : REF_PIC_LIST_0);
      if( pcCU->getSlice()->getUseIntegerMv() )
      {
        cMvd.setHor( (iMvCandList[2*refIdxOpt + 4*refListOpt].getHor()>>2) - (cMvPredCand[1][0].getHor()>>2) );
        cMvd.setVer( (iMvCandList[2*refIdxOpt + 4*refListOpt].getVer()>>2) - (cMvPredCand[1][0].getVer()>>2) );
        pcCU->getCUMvField( eRefPicListOpt )->setAllMv( (iMvCandList[2*refIdxOpt + 4*refListOpt]>>2)<<2, ePartSize, partAddr, 0, partIdx );
      }
      else
      {
        cMvd.setHor( iMvCandList[2*refIdxOpt + 4*refListOpt].getHor() - cMvPredCand[1][0].getHor() );
        cMvd.setVer( iMvCandList[2*refIdxOpt + 4*refListOpt].getVer() - cMvPredCand[1][0].getVer() );
        pcCU->getCUMvField( eRefPicListOpt )->setAllMv( iMvCandList[2*refIdxOpt + 4*refListOpt], ePartSize, partAddr, 0, partIdx );
      }
      pcCU->getCUMvField( eRefPicListOpt )->setAllMvd( cMvd, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( eRefPicListOpt )->setAllRefIdx( refIdxOpt, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( (RefPicList)(1-refListOpt) )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
      pcCU->setInterDirSubParts( 1 + refListOpt, partAddr, partIdx, depth );
      pcCU->setMergeFlagSubParts( false, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( bestInterMvpIdx[1], eRefPicListOpt, partAddr, partIdx, depth );
    }
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif

    partIdx = 1;
    pcCU->getPartIndexAndSize( partIdx, partAddr, dummyWidth2, dummyHeight2 );

    if ( isIBCMergeMode[1] )
    {
      pcCU->setMergeFlagSubParts( true, partAddr, partIdx, depth );
      pcCU->setMergeIndexSubParts( bestIBCMvpIdx[1], partAddr, partIdx, depth );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMRGMvFieldIBC[1][0], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMRGMvFieldIBC[1][1], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, depth );

    }
    else
    {
      pcCU->setMergeFlagSubParts( false, partAddr, partIdx, depth );
      cMvd.setHor( (iMvCandList[9].getHor() - cMvPredCand[1][1].getHor())>>2 );
      cMvd.setVer( (iMvCandList[9].getVer() - cMvPredCand[1][1].getVer())>>2 );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( iMvCandList[9], ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvd, ePartSize, partAddr, 0, partIdx );
      pcCU->setMVPIdxSubParts( bestIBCMvpIdx[1], REF_PIC_LIST_0, partAddr, partIdx, depth );
      pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, ePartSize, partAddr, 0, partIdx );
      pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, ePartSize, partAddr, 0, partIdx );
      pcCU->setInterDirSubParts( 1, partAddr, partIdx, depth );  // list 0 prediction
    }
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif
  }
  motionCompensation( pcCU, rpcPredYuv );

  return true;
}

Void TEncSearch::addToSortList( list<BlockHash>& listBlockHash, list<Int>& listCost, Int cost, const BlockHash& blockHash )
{
  assert( listBlockHash.size() == listCost.size() );
  list<BlockHash>::iterator itBlockHash = listBlockHash.begin();
  list<Int>::iterator itCost = listCost.begin();

  while ( itCost != listCost.end() )
  {
    if ( cost < (*itCost) )
    {
      listCost.insert( itCost, cost );
      listBlockHash.insert( itBlockHash, blockHash );
      return;
    }

    ++itCost;
    ++itBlockHash;
  }

  listCost.push_back( cost );
  listBlockHash.push_back( blockHash );
}

Distortion TEncSearch::getSAD( Pel* pRef, Int refStride, Pel* pCurr, Int currStride, Int width, Int height, const BitDepths& bitDepths )
{
  Distortion dist = 0;

  for ( Int i=0; i<height; i++ )
  {
    for ( Int j=0; j<width; j++ )
    {
      dist += abs( pRef[j] - pCurr[j] );
    }
    pRef += refStride;
    pCurr += currStride;
  }

  if ( bitDepths.recon[CHANNEL_TYPE_LUMA] == 8 )
  {
    return dist;
  }
  else
  {
    Int shift = DISTORTION_PRECISION_ADJUSTMENT( bitDepths.recon[CHANNEL_TYPE_LUMA] - 8 );
    return dist >> shift;
  }
  return 0;
}

Bool TEncSearch::predInterHashSearch( TComDataCU* pcCU, TComYuv*& rpcPredYuv, Bool& isPerfectMatch )
{
  rpcPredYuv->clear();
  TComMvField  cMEMvField;
  TComMv       bestMv, bestMvd;
  RefPicList   bestRefPicList;
  Int          bestRefIndex;
  Int          bestMVPIndex;
  Int          partIdx   = 0;
  Int          partAddr = 0;
  PartSize     ePartSize  = pcCU->getPartitionSize( 0 );

  //  Clear Motion Field
  TComMv cMvZero( 0, 0 );
  pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( TComMvField(), ePartSize, partAddr, 0, partIdx );
  pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( TComMvField(), ePartSize, partAddr, 0, partIdx );
  pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );
  pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvd( cMvZero, ePartSize, partAddr, 0, partIdx );

  pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth( partAddr ) );
  pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_0, partAddr, partIdx, pcCU->getDepth( partAddr ) );
  pcCU->setMVPIdxSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, pcCU->getDepth( partAddr ) );
  pcCU->setMVPNumSubParts( -1, REF_PIC_LIST_1, partAddr, partIdx, pcCU->getDepth( partAddr ) );

  if ( xHashInterEstimation( pcCU, pcCU->getWidth( 0 ), pcCU->getHeight( 0 ), bestRefPicList, bestRefIndex, bestMv, bestMvd, bestMVPIndex, isPerfectMatch ) )
  {
    pcCU->getCUMvField( bestRefPicList )->setAllMv( bestMv, ePartSize, partAddr, 0, partIdx );
    pcCU->getCUMvField( bestRefPicList )->setAllRefIdx( bestRefIndex, ePartSize, partAddr, 0, partIdx );
    pcCU->getCUMvField( bestRefPicList )->setAllMvd( bestMvd, ePartSize, partAddr, 0, partIdx );

    pcCU->setInterDirSubParts( static_cast<Int>(bestRefPicList) + 1, partAddr, partIdx, pcCU->getDepth( 0 ) );
    pcCU->setMVPIdxSubParts( bestMVPIndex, bestRefPicList, partAddr, partIdx, pcCU->getDepth( partAddr ) );

#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif

    motionCompensation ( pcCU, rpcPredYuv, REF_PIC_LIST_X, partIdx );
    return true;
  }
  else
  {
    return false;
  }

  assert( 0 );
  return true;
}

Void TEncSearch::selectMatchesInter( const MapIterator& itBegin, Int count, list<BlockHash>& listBlockHash, const BlockHash& currBlockHash )
{
  const Int maxReturnNumber = 5;

  listBlockHash.clear();
  list<Int> listCost;
  listCost.clear();

  MapIterator it = itBegin;
  for ( Int i=0; i<count; i++, it++ )
  {
    if ( (*it).hashValue2 != currBlockHash.hashValue2 )  // check having the same second hash values, otherwise, not matched
    {
      continue;
    }

    // as exactly matched, only calculate bits
    Int currCost = TComRdCost::xGetExpGolombNumberOfBits( (*it).x - currBlockHash.x ) +
                   TComRdCost::xGetExpGolombNumberOfBits( (*it).y - currBlockHash.y );

    if ( listBlockHash.size() < maxReturnNumber )
    {
      addToSortList( listBlockHash, listCost, currCost, (*it) );
    }
    else if ( !listCost.empty() && currCost < listCost.back( ) )
    {
      listCost.pop_back( );
      listBlockHash.pop_back();
      addToSortList( listBlockHash, listCost, currCost, (*it) );
    }
  }
}

Int TEncSearch::xHashInterPredME( const TComDataCU* const pcCU, Int width, Int height, RefPicList currRefPicList, Int currRefPicIndex, TComMv bestMv[5] )
{
  const TComPic* pcPic = pcCU->getPic();
  Int xPos = pcCU->getCUPelX();
  Int yPos = pcCU->getCUPelY();
  UInt hashValue1;
  UInt hashValue2;

  if ( !TComHash::getBlockHashValue( pcPic->getPicYuvOrg(), width, height, xPos, yPos, pcCU->getSlice()->getSPS()->getBitDepths(), hashValue1, hashValue2 ) )
  {
    return 0;
  }


  BlockHash currBlockHash;
  currBlockHash.x = xPos;
  currBlockHash.y = yPos;
  currBlockHash.hashValue2 = hashValue2;

  Int count = static_cast<Int>(pcCU->getSlice()->getRefPic( currRefPicList, currRefPicIndex )->getHashMap()->count( hashValue1 ));
  if ( count == 0 )
  {
    return 0;
  }

  list<BlockHash> listBlockHash;
  selectMatchesInter( pcCU->getSlice()->getRefPic( currRefPicList, currRefPicIndex )->getHashMap()->getFirstIterator( hashValue1 ), count, listBlockHash, currBlockHash );

  if ( listBlockHash.empty() )
  {
    return 0;
  }

  Int totalSize = 0;
  list<BlockHash>::iterator it = listBlockHash.begin();
  for ( Int i=0; i<5 && i<listBlockHash.size(); i++, it++ )
  {
    bestMv[i].set( (*it).x - currBlockHash.x, (*it).y - currBlockHash.y );
    totalSize++;
  }

  return totalSize;
}

Bool TEncSearch::xHashInterEstimation( TComDataCU* pcCU, Int width, Int height, RefPicList& bestRefPicList, Int& bestRefIndex, TComMv& bestMv, TComMv& bestMvd, Int& bestMVPIndex, Bool& isPerfectMatch )
{
  TComPic* pcPic = pcCU->getPic();
  Int xPos = pcCU->getCUPelX();
  Int yPos = pcCU->getCUPelY();
  UInt hashValue1;
  UInt hashValue2;
  Int bestCost = MAX_INT;

  if ( !TComHash::getBlockHashValue( pcPic->getPicYuvOrg(), width, height, xPos, yPos, pcCU->getSlice()->getSPS()->getBitDepths(), hashValue1, hashValue2 ) )
  {
    return false;
  }

  BlockHash currBlockHash;
  currBlockHash.x = xPos;
  currBlockHash.y = yPos;
  currBlockHash.hashValue2 = hashValue2;

  Pel* pCurrStart = pcCU->getPic()->getPicYuvOrg()->getAddr( COMPONENT_Y );
  Int currStride = pcCU->getPic()->getPicYuvOrg()->getStride( COMPONENT_Y );
  Pel* pCurr = pCurrStart + (currBlockHash.y)*currStride + (currBlockHash.x);

  Int numPredDir = pcCU->getSlice()->isInterP() ? 1 : 2;
  for ( Int refList = 0; refList < numPredDir; refList++ )
  {
    RefPicList eRefPicList = (refList==0) ? REF_PIC_LIST_0 : REF_PIC_LIST_1;
    Int refPicNumber = pcCU->getSlice()->getNumRefIdx( eRefPicList );
    if ( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && eRefPicList == REF_PIC_LIST_0 )
    {
      refPicNumber--;
    }
    for ( Int refIdx = 0; refIdx < refPicNumber; refIdx++ )
    {
      Int bitsOnRefIdx = refIdx+1;
      if ( refIdx+1 == pcCU->getSlice()->getNumRefIdx( eRefPicList ) )
      {
        bitsOnRefIdx--;
      }

      if ( refList == 0 || pcCU->getSlice()->getList1IdxToList0Idx( refIdx ) < 0 )
      {
        Int count = static_cast<Int>( pcCU->getSlice()->getRefPic( eRefPicList, refIdx )->getHashMap()->count( hashValue1 ) );
        if ( count == 0 )
        {
          continue;
        }

        list<BlockHash> listBlockHash;
        selectMatchesInter( pcCU->getSlice()->getRefPic( eRefPicList, refIdx )->getHashMap()->getFirstIterator( hashValue1 ), count, listBlockHash, currBlockHash );

        if ( listBlockHash.empty() )
        {
          continue;
        }

        AMVPInfo currAMVPInfo;
        pcCU->fillMvpCand( 0, 0, eRefPicList, refIdx, &currAMVPInfo );

        Pel* pRefStart = pcCU->getSlice()->getRefPic( eRefPicList, refIdx )->getPicYuvRec()->getAddr( COMPONENT_Y );
        Int refStride = pcCU->getSlice()->getRefPic( eRefPicList, refIdx )->getPicYuvRec()->getStride( COMPONENT_Y );

        m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass( 0 ) );
        m_pcRdCost->setCostScale( 2 );
        if ( pcCU->getSlice()->getUseIntegerMv() )
        {
          m_pcRdCost->setCostScale( 0 );
        }
        list<BlockHash>::iterator it;
        for ( it = listBlockHash.begin(); it != listBlockHash.end(); ++it )
        {
          Int currMVPIdx = 0;
          if ( currAMVPInfo.iN > 1 )
          {
            m_pcRdCost->setPredictor( currAMVPInfo.m_acMvCand[0] );
            Int bitsMVP0 = m_pcRdCost->getBitsOfVectorWithPredictor( (*it).x - currBlockHash.x, (*it).y - currBlockHash.y );
            m_pcRdCost->setPredictor( currAMVPInfo.m_acMvCand[1] );
            Int bitsMVP1 = m_pcRdCost->getBitsOfVectorWithPredictor( (*it).x - currBlockHash.x, (*it).y - currBlockHash.y );
            if ( bitsMVP1 < bitsMVP0 )
            {
              currMVPIdx = 1;
            }
          }
          m_pcRdCost->setPredictor( currAMVPInfo.m_acMvCand[currMVPIdx] );

          Pel* pRef = pRefStart + (*it).y*refStride + (*it).x;
          Distortion currSad = getSAD( pRef, refStride, pCurr, currStride, width, height, pcCU->getSlice()->getSPS()->getBitDepths() );
          Int bits = bitsOnRefIdx + m_pcRdCost->getBitsOfVectorWithPredictor( (*it).x - currBlockHash.x, (*it).y - currBlockHash.y );
          Distortion currCost = currSad + m_pcRdCost->getCost( bits );

          if ( !isPerfectMatch && pcCU->getPartitionSize( 0 ) == SIZE_2Nx2N )
          {
            if ( pcCU->getSlice()->getRefPic( eRefPicList, refIdx )->getSlice( 0 )->getSliceQp() <= pcCU->getSlice()->getSliceQp() )
            {
              isPerfectMatch = true;
            }
          }

          if ( currCost < bestCost )
          {
            bestCost = (Int)currCost;
            bestRefPicList = eRefPicList;
            bestRefIndex = refIdx;
            bestMv.set( (*it).x - currBlockHash.x, (*it).y - currBlockHash.y );
            bestMv <<= 2;

            if( pcCU->getSlice()->getUseIntegerMv() )
            {
              bestMvd.set( (bestMv.getHor()>>2) - (currAMVPInfo.m_acMvCand[currMVPIdx].getHor()>>2), (bestMv.getVer()>>2) - (currAMVPInfo.m_acMvCand[currMVPIdx].getVer()>>2) );
            }
            else
            {
               bestMvd.set( bestMv.getHor() - currAMVPInfo.m_acMvCand[currMVPIdx].getHor(), bestMv.getVer() - currAMVPInfo.m_acMvCand[currMVPIdx].getVer() );
            }
            bestMVPIndex = currMVPIdx;
          }
        }
      }
    }
  }

  return (bestCost < MAX_INT);
}

// based on xMotionEstimation
Void TEncSearch::xIntraBlockCopyEstimation( TComDataCU *pcCU,
                                            TComYuv    *pcYuvOrg,
                                            Int         iPartIdx,
                                            TComMv     *pcMvPred,
                                            TComMv     &rcMv,
                                            Distortion &cost,
                                            Bool        bUse1DSearchFor8x8,
                                            Bool        testOnlyPred
                                           )
{
  UInt          partAddr;
  Int           roiWidth;
  Int           roiHeight;

  TComMv        cMvSrchRngLT;
  TComMv        cMvSrchRngRB;

  TComYuv*      pcYuv = pcYuvOrg;

  TComPattern   cPattern;

  pcCU->getPartIndexAndSize( iPartIdx, partAddr, roiWidth, roiHeight );

  //  Search key pattern initialization
#if MCTS_ENC_CHECK
  Int roiPosX, roiPosY;
  Int roiW, roiH;
  pcCU->getPartPosition(iPartIdx, roiPosX, roiPosY, roiW, roiH);
  assert(roiW == roiWidth);
  assert(roiH == roiHeight);
  cPattern.initPattern( pcYuv->getAddr(COMPONENT_Y, partAddr),
                        roiWidth,
                        roiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA),
                        roiPosX,
                        roiPosY);
  xInitTileBorders(pcCU, &cPattern);
#else
  cPattern.initPattern( pcYuv->getAddr  ( COMPONENT_Y, partAddr ),
                        roiWidth,
                        roiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
#endif

  Pel*        piRefY      = pcCU->getPic()->getPicYuvRec()->getAddr( COMPONENT_Y, pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + partAddr );
  Int         refStride   = pcCU->getPic()->getPicYuvRec()->getStride(COMPONENT_Y);

  TComMv      cMvPred = *pcMvPred;

  // assume that intra BV is integer-pel precision
  xSetIntraSearchRange   ( pcCU, cMvPred, partAddr, roiWidth, roiHeight, cMvSrchRngLT, cMvSrchRngRB );

  // disable weighted prediction
  setWpScalingDistParam( pcCU, -1, REF_PIC_LIST_X );

  m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(partAddr) );
  m_pcRdCost->setPredictors(pcMvPred);
  m_pcRdCost->setCostScale  ( 0 );

  //  Do integer search
  xIntraPatternSearch( pcCU, iPartIdx, partAddr, &cPattern, piRefY, refStride, &cMvSrchRngLT, &cMvSrchRngRB, rcMv, cost, roiWidth, roiHeight, bUse1DSearchFor8x8, testOnlyPred );
  //printf("cost = %d\n", cost);
}

// based on xSetSearchRange
Void TEncSearch::xSetIntraSearchRange ( TComDataCU* pcCU, TComMv& cMvPred, UInt partAddr, Int roiWidth, Int roiHeight, TComMv& rcMvSrchRngLT, TComMv& rcMvSrchRngRB )
{
  TComMv cTmpMvPred = cMvPred;
  pcCU->clipMv( cTmpMvPred );

  Int srLeft, srRight, srTop, srBottom;

  const UInt lcuWidth = pcCU->getSlice()->getSPS()->getMaxCUWidth();
  const UInt cuPelX   = pcCU->getCUPelX() + g_auiRasterToPelX[ g_auiZscanToRaster[ partAddr ] ]; //NOTE: RExt - This variable (and its counterpart below) refer to the PU, not the CU - change these names
  const UInt lcuHeight = pcCU->getSlice()->getSPS()->getMaxCUHeight();
  const UInt cuPelY    = pcCU->getCUPelY() + g_auiRasterToPelY[ g_auiZscanToRaster[ partAddr ] ];

  const Int picWidth  = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
  const Int picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();

  const UInt curTileIdx = pcCU->getPic()->getPicSym()->getTileIdxMap( pcCU->getCtuRsAddr() );
  TComTile* curTile = pcCU->getPic()->getPicSym()->getTComTile( curTileIdx );

  const Int tileAreaRight  = (curTile->getRightEdgePosInCtus() + 1) * lcuWidth;
  const Int tileAreaBottom = (curTile->getBottomEdgePosInCtus() + 1) * lcuHeight;

  const Int tileAreaLeft   = tileAreaRight - curTile->getTileWidthInCtus() * lcuWidth;
  const Int tileAreaTop    = tileAreaBottom - curTile->getTileHeightInCtus() * lcuHeight;

  if((pcCU->getWidth(0) == 16) && (pcCU->getPartitionSize(0) == SIZE_2Nx2N) && m_pcEncCfg->getUseIntraBCFullFrameSearch())
  {
    srLeft  = -1 * cuPelX;
    srTop   = -1 * cuPelY;
    TComSlice *pcSlice = pcCU->getSlice();
    if( pcSlice->getSliceMode() )
    {
      TComPicSym *pcSym = pcCU->getPic()->getPicSym();
      UInt addr = pcSym->getCtuTsToRsAddrMap( pcSlice->getSliceSegmentCurStartCtuTsAddr() );
      srTop += pcSym->getCtu(addr)->getCUPelY();
    }

    srRight = picWidth - cuPelX - roiWidth;
    srBottom = lcuHeight - cuPelY % lcuHeight - roiHeight;
  }
  else
  {
    const UInt searchWidthInCTUs = pcCU->getWidth( 0 ) == 8 ? m_pcEncCfg->getIntraBCNonHashSearchWidthInCTUs() : m_pcEncCfg->getIntraBCSearchWidthInCTUs();
    Int maxXsr = (cuPelX % lcuWidth) + pcCU->getIntraBCSearchAreaWidth( searchWidthInCTUs );
    Int maxYsr =  cuPelY % lcuHeight;

    const ChromaFormat format = pcCU->getPic()->getChromaFormat();

    if ((format == CHROMA_420) || (format == CHROMA_422)) maxXsr &= ~0x4;
    if ((format == CHROMA_420)                          ) maxYsr &= ~0x4;

    srLeft   = -maxXsr;
    srTop    = -maxYsr;

    srRight = lcuWidth - cuPelX %lcuWidth - roiWidth;
    srBottom = lcuHeight - cuPelY % lcuHeight - roiHeight;
  }


  if( cuPelX + srRight + roiWidth > picWidth)
  {
    srRight = picWidth%lcuWidth - cuPelX %lcuWidth - roiWidth;
  }
  if( cuPelY + srBottom + roiHeight > picHeight)
  {
    srBottom = picHeight%lcuHeight - cuPelY % lcuHeight - roiHeight;
  }

  if( cuPelX + srRight + roiWidth > tileAreaRight)
  {
    srRight = tileAreaRight%lcuWidth - cuPelX %lcuWidth - roiWidth;
  }
  if( cuPelY + srBottom + roiHeight > tileAreaBottom)
  {
    srBottom = tileAreaBottom%lcuHeight - cuPelY % lcuHeight - roiHeight;
  }
  if( cuPelX + srLeft < tileAreaLeft)
  {
    srLeft = tileAreaLeft - cuPelX;
  }
  if( cuPelY + srTop < tileAreaTop)
  {
    srTop = tileAreaTop - cuPelY;
  }

  rcMvSrchRngLT.setHor( srLeft );
  rcMvSrchRngLT.setVer( srTop );
  rcMvSrchRngRB.setHor( srRight );
  rcMvSrchRngRB.setVer( srBottom );

  pcCU->clipMv        ( rcMvSrchRngLT );
  pcCU->clipMv        ( rcMvSrchRngRB );
}

Void TEncSearch::xIntraBCSearchMVCandUpdate(Distortion sad, Int x, Int y, Distortion* sadBestCand, TComMv* cMVCand)
{
  int j = CHROMA_REFINEMENT_CANDIDATES - 1;

  if(sad < sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
  {
    for(int t = CHROMA_REFINEMENT_CANDIDATES - 1; t >= 0; t--)
    {
      if (sad < sadBestCand[t])
      {
        j = t;
      }
    }

    for(int k = CHROMA_REFINEMENT_CANDIDATES - 1;k > j; k--)
    {
      sadBestCand[k]=sadBestCand[k-1];

      cMVCand[k].set(cMVCand[k-1].getHor(),cMVCand[k-1].getVer());
    }
    sadBestCand[j]= sad;
    cMVCand[j].set(x,y);
  }
}

Int TEncSearch::xIntraBCSearchMVChromaRefine( TComDataCU* pcCU,
                                              Int         roiWidth,
                                              Int         roiHeight,
                                              Int         cuPelX,
                                              Int         cuPelY,
                                              Distortion* sadBestCand,
                                              TComMv*     cMVCand,
                                              UInt        partOffset,
                                              Int         partIdx
                                            )
{
  Int bestCandIdx = 0;
  Distortion  sadBest = std::numeric_limits<Distortion>::max();
  Distortion  tempSad;

  Pel* pRef;
  Pel* pOrg;
  Int refStride, orgStride;
  Int width, height;

  Int picWidth = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
  Int picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();

  for(int cand = 0; cand < CHROMA_REFINEMENT_CANDIDATES; cand++)
  {
    if ((!cMVCand[cand].getHor()) && (!cMVCand[cand].getVer()))
    {
      continue;
    }

    if (((Int)(cuPelY + cMVCand[cand].getVer() + roiHeight) >= picHeight) || ((cuPelY + cMVCand[cand].getVer()) < 0))
    {
      continue;
    }

    if (((Int)(cuPelX + cMVCand[cand].getHor() + roiWidth) >= picWidth) || ((cuPelX + cMVCand[cand].getHor()) < 0))
    {
      continue;
    }

    tempSad = sadBestCand[cand];
    BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
    TComMv cMvQuaterPixl = cMVCand[cand];
    cMvQuaterPixl <<= 2;
    pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllMv( cMvQuaterPixl, pcCU->getPartitionSize(0), partOffset, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_0 )->setAllRefIdx( pcCU->getSlice()->getNumRefIdx( REF_PIC_LIST_0 ) - 1, pcCU->getPartitionSize(0), partOffset, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllMv( TComMv(), pcCU->getPartitionSize(0), partOffset, 0, partIdx );
    pcCU->getCUMvField( REF_PIC_LIST_1 )->setAllRefIdx( -1, pcCU->getPartitionSize(0), partOffset, 0, partIdx );
    pcCU->setInterDirSubParts( 1 + REF_PIC_LIST_0, partOffset, partIdx, pcCU->getDepth(0) );
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!checkTMctsMvp(pcCU, partIdx)))
    {
      pcCU->setTMctsMvpIsValid(false);
      return false;
    }
#endif
    motionCompensation( pcCU, &m_tmpYuvPred, REF_PIC_LIST_X, partIdx );

    for (UInt ch = COMPONENT_Cb; ch < pcCU->getPic()->getNumberValidComponents(); ch++)
    {
      pRef = pcCU->getPic()->getPicYuvRec()->getAddr(ComponentID(ch), pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + partOffset);
      if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && m_pcEncCfg->getRGBFormatFlag() )
      {
        pOrg = pcCU->getPic()->getPicYuvResi()->getAddr(ComponentID(ch), pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + partOffset);
      }
      else
      {
        pOrg = pcCU->getPic()->getPicYuvOrg()->getAddr(ComponentID(ch), pcCU->getCtuRsAddr(), pcCU->getZorderIdxInCtu() + partOffset);
      }
      refStride = pcCU->getPic()->getPicYuvRec()->getStride(ComponentID(ch));
      if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && m_pcEncCfg->getRGBFormatFlag() )
      {
        orgStride = pcCU->getPic()->getPicYuvResi()->getStride(ComponentID(ch));
      }
      else
      {
        orgStride = pcCU->getPic()->getPicYuvOrg()->getStride(ComponentID(ch));
      }
      width = roiWidth >> pcCU->getPic()->getComponentScaleX(ComponentID(ch));
      height = roiHeight >> pcCU->getPic()->getComponentScaleY(ComponentID(ch));

      ComponentID compID = (ComponentID)ch;
      pRef = m_tmpYuvPred.getAddr( compID, partOffset);
      refStride = m_tmpYuvPred.getStride(compID);

      for(int row = 0; row < height; row++)
      {
        for(int col = 0; col < width; col++)
        {
          tempSad += ( (abs( pRef[col] - pOrg[col] )) >> (bitDepths.recon[CHANNEL_TYPE_CHROMA]-8) );
        }
        pRef += refStride;
        pOrg += orgStride;
      }
    }

    if(tempSad < sadBest)
    {
      sadBest = tempSad;
      bestCandIdx = cand;
    }
  }

  return bestCandIdx;
}

static UInt MergeCandLists(TComMv *dst, UInt dn, TComMv *src, UInt sn, Bool isSrcQuarPel)
{
  for(UInt cand = 0; cand < sn && dn<SCM_S0067_NUM_CANDIDATES; cand++)
  {
    Bool found = false;
    TComMv TempMv = src[cand];
    if ( !isSrcQuarPel )
    {
      TempMv <<= 2;
    }
    for(int j=0; j<dn; j++)
    {
      if( TempMv == dst[j] )
      {
        found = true;
        break;
      }
    }

    if( !found )
    {
      dst[dn] = TempMv;
      dn++;
    }
  }

  return dn;
}

// based on xPatternSearch
Void TEncSearch::xIntraPatternSearch( TComDataCU  *pcCU,
                                      Int          partIdx,
                                      UInt         partAddr,
                                      TComPattern *pcPatternKey,
                                      Pel         *piRefY,
                                      Int          refStride,
                                      TComMv      *pcMvSrchRngLT,
                                      TComMv      *pcMvSrchRngRB,
                                      TComMv      &rcMv,
                                      Distortion  &SAD,
                                      Int          roiWidth,
                                      Int          roiHeight,
                                      Bool         bUse1DSearchFor8x8,
                                      Bool         testOnlyPred
                                      )
{
  const Int   srchRngHorLeft   = pcMvSrchRngLT->getHor();
  const Int   srchRngHorRight  = pcMvSrchRngRB->getHor();
  const Int   srchRngVerTop    = pcMvSrchRngLT->getVer();
  const Int   srchRngVerBottom = pcMvSrchRngRB->getVer();

  const UInt  lcuWidth         = pcCU->getSlice()->getSPS()->getMaxCUWidth();
  const UInt  lcuHeight        = pcCU->getSlice()->getSPS()->getMaxCUHeight();
  const Int   puPelOffsetX     = g_auiRasterToPelX[ g_auiZscanToRaster[ partAddr ] ];
  const Int   puPelOffsetY     = g_auiRasterToPelY[ g_auiZscanToRaster[ partAddr ] ];
  const Int   cuPelX           = pcCU->getCUPelX() + puPelOffsetX;  // Point to the location of PU
  const Int   cuPelY           = pcCU->getCUPelY() + puPelOffsetY;

  Distortion  sad;
  Distortion  sadBest          = std::numeric_limits<Distortion>::max();
  Int         bestX            = 0;
  Int         bestY            = 0;

  Pel*        piRefSrch;

  Int         bestCandIdx = 0;
  UInt        partOffset = 0;
  Distortion  sadBestCand[CHROMA_REFINEMENT_CANDIDATES];
  TComMv      cMVCand[CHROMA_REFINEMENT_CANDIDATES];

  partOffset = partAddr;

  for(int cand = 0; cand < CHROMA_REFINEMENT_CANDIDATES; cand++)
  {
    sadBestCand[cand] = std::numeric_limits<Distortion>::max();
    cMVCand[cand].set(0,0);
  }

  //-- jclee for using the SAD function pointer
  m_pcRdCost->setDistParam( pcPatternKey, piRefY, refStride,  m_cDistParam );

  const Int        relCUPelX    = cuPelX % lcuWidth;
  const Int        relCUPelY    = cuPelY % lcuHeight;
  const Int chromaROIWidthInPixels  = roiWidth;
  const Int chromaROIHeightInPixels = roiHeight;

  const UInt curTileIdx = pcCU->getPic()->getPicSym()->getTileIdxMap( pcCU->getCtuRsAddr() );
  TComTile* curTile = pcCU->getPic()->getPicSym()->getTComTile( curTileIdx );

  const Int tileAreaRight  = (curTile->getRightEdgePosInCtus() + 1) * lcuWidth;
  const Int tileAreaBottom = (curTile->getBottomEdgePosInCtus() + 1) * lcuHeight;

  const Int tileAreaLeft   = tileAreaRight - curTile->getTileWidthInCtus() * lcuWidth;
  const Int tileAreaTop    = tileAreaBottom - curTile->getTileHeightInCtus() * lcuHeight;

  if (m_pcEncCfg->getUseIntraBlockCopyFastSearch())
  {
    setDistParamComp(COMPONENT_Y);
    m_cDistParam.bitDepth  = pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA );
    m_cDistParam.iRows     = 4;//to calculate the sad line by line;
    m_cDistParam.iSubShift = 0;

    Distortion tempSadBest = 0;

    Int srLeft = srchRngHorLeft, srRight = srchRngHorRight, srTop = srchRngVerTop, srBottom = srchRngVerBottom;

    const Int picWidth  = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
    const Int picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();

    if(m_pcEncCfg->getUseIntraBCFullFrameSearch() )
    {
      srLeft  = -1 * cuPelX;
      srTop   = -1 * cuPelY;

      srRight  = picWidth - cuPelX - roiWidth;
      srBottom = lcuHeight - cuPelY % lcuHeight - roiHeight;

      if( cuPelX + srRight + roiWidth > picWidth)
      {
        srRight = picWidth%lcuWidth - cuPelX %lcuWidth - roiWidth;
      }
      if( cuPelY + srBottom + roiHeight > picHeight)
      {
        srBottom = picHeight%lcuHeight - cuPelY % lcuHeight - roiHeight;
      }
      if( cuPelX + srRight + roiWidth > tileAreaRight)
      {
        srRight = tileAreaRight%lcuWidth - cuPelX %lcuWidth - roiWidth;
      }
      if( cuPelY + srBottom + roiHeight > tileAreaBottom)
      {
        srBottom = tileAreaBottom%lcuHeight - cuPelY % lcuHeight - roiHeight;
      }
      if( cuPelX + srLeft < tileAreaLeft)
      {
        srLeft = tileAreaLeft - cuPelX;
      }
      if( cuPelY + srTop < tileAreaTop)
      {
        srTop = tileAreaTop - cuPelY;
      }
    }

    if(roiWidth>8 || roiHeight>8)
    {
      m_numBVs = 0;
    }
    else if (roiWidth+roiHeight==16)
    {
      m_numBVs = m_numBV16s;
    }

    if( testOnlyPred )
    {
      m_numBVs = 0;
    }

    TComMv cMvPredEncOnly[16];
    Int nbPreds = 0;
    pcCU->getIntraBCMVPsEncOnly(partAddr, cMvPredEncOnly, nbPreds, partIdx );
    m_numBVs = MergeCandLists(m_acBVs, m_numBVs, cMvPredEncOnly, nbPreds, true);
    for(UInt cand = 0; cand < m_numBVs; cand++)
    {
      Int xPred = m_acBVs[cand].getHor()>>2;
      Int yPred = m_acBVs[cand].getVer()>>2;
      if ( !( xPred==0 && yPred==0)
            && !( (yPred < srTop)  || (yPred > srBottom) )
            && !( (xPred < srLeft) || (xPred > srRight) ) )
      {
        Int tempY = yPred + relCUPelY + roiHeight - 1;
        Int tempX = xPred + relCUPelX + roiWidth  - 1;
        Bool validCand = isValidIntraBCSearchArea(pcCU, xPred, yPred, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset);

        if((tempX >= (Int)lcuWidth) && (tempY >= 0) && m_pcEncCfg->getUseIntraBCFullFrameSearch())
        {
          validCand = false;
        }

        if ((tempX >= 0) && (tempY >= 0))
        {
          Int iTempRasterIdx = (tempY/pcCU->getPic()->getMinCUHeight()) * pcCU->getPic()->getNumPartInCtuWidth() + (tempX/pcCU->getPic()->getMinCUWidth());
          Int iTempZscanIdx = g_auiRasterToZscan[iTempRasterIdx];
          if(iTempZscanIdx >= pcCU->getZorderIdxInCtu())
          {
            validCand = false;
          }
        }

        if( validCand )
        {
          sad = m_pcRdCost->getCostMultiplePreds( xPred, yPred);

          for(int r = 0; r < roiHeight; )
          {
            piRefSrch = piRefY + yPred * refStride + r*refStride + xPred;
            m_cDistParam.pCur = piRefSrch;
            m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

            sad += m_cDistParam.DistFunc( &m_cDistParam );
            if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
            {
              break;
            }

            r += 4;
          }

          xIntraBCSearchMVCandUpdate(sad, xPred, yPred, sadBestCand, cMVCand);
        }
      }
    }

    bestX = cMVCand[0].getHor();
    bestY = cMVCand[0].getVer();
    rcMv.set( bestX, bestY );
    sadBest = sadBestCand[0];

    if( testOnlyPred )
    {
      SAD = sadBest;
      return;
    }

    const Int boundY = (0 - roiHeight - puPelOffsetY);
    Int lowY = ((pcCU->getPartitionSize(partAddr) == SCM_S0067_IBC_FULL_1D_SEARCH_FOR_PU) && m_pcEncCfg->getUseIntraBCFullFrameSearch())
             ? -cuPelY : max(srchRngVerTop, 0 - cuPelY);
    for(Int y = boundY ; y >= lowY ; y-- )
    {
      if ( !isValidIntraBCSearchArea( pcCU, 0, y, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset ) )
      {
        continue;
      }

      sad = m_pcRdCost->getCostMultiplePreds( 0, y);

      for(int r = 0; r < roiHeight; )
      {
        piRefSrch = piRefY + y * refStride + r*refStride;
        m_cDistParam.pCur = piRefSrch;
        m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

        sad += m_cDistParam.DistFunc( &m_cDistParam );

        if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
        {
          break;
        }

        r += 4;
      }

      xIntraBCSearchMVCandUpdate(sad, 0, y, sadBestCand, cMVCand);
      tempSadBest = sadBestCand[0];
      if(sadBestCand[0] <= 3)
      {
        bestX = cMVCand[0].getHor();
        bestY = cMVCand[0].getVer();
        sadBest = sadBestCand[0];
        rcMv.set( bestX, bestY );
        SAD = sadBest;

        updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
        return;
      }
    }

    const Int boundX = ((pcCU->getPartitionSize(partAddr) == SCM_S0067_IBC_FULL_1D_SEARCH_FOR_PU) && m_pcEncCfg->getUseIntraBCFullFrameSearch())
                     ? -cuPelX : max(srchRngHorLeft, - cuPelX);
    for(Int x = 0 - roiWidth - puPelOffsetX ; x >= boundX ; --x )
    {
      if (!isValidIntraBCSearchArea(pcCU, x, 0, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset))
      {
        continue;
      }

      sad = m_pcRdCost->getCostMultiplePreds( x, 0);

      for(int r = 0; r < roiHeight; )
      {
        piRefSrch = piRefY + r*refStride + x;
        m_cDistParam.pCur = piRefSrch;
        m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

        sad += m_cDistParam.DistFunc( &m_cDistParam );
        if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
        {
          break;
        }

        r += 4;
      }

      xIntraBCSearchMVCandUpdate(sad, x, 0, sadBestCand, cMVCand);
      tempSadBest = sadBestCand[0];
      if(sadBestCand[0] <= 3)
      {
        bestX = cMVCand[0].getHor();
        bestY = cMVCand[0].getVer();
        sadBest = sadBestCand[0];
        rcMv.set( bestX, bestY );
        SAD = sadBest;

        updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
        return;
      }
    }

    bestX = cMVCand[0].getHor();
    bestY = cMVCand[0].getVer();
    sadBest = sadBestCand[0];
    if((!bestX && !bestY) || (sadBest - m_pcRdCost->getCostMultiplePreds( bestX, bestY) <= 32))
    {
      //chroma refine
      bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
      bestX       = cMVCand[bestCandIdx].getHor();
      bestY       = cMVCand[bestCandIdx].getVer();
      sadBest    = sadBestCand[bestCandIdx];
      rcMv.set( bestX, bestY );
      SAD       = sadBest;

      updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
      return;
    }

    if( pcCU->getWidth(0) < 16 && !bUse1DSearchFor8x8 )
    {
      for(Int y = max(srchRngVerTop, -cuPelY); y <= srchRngVerBottom; y +=2)
      {
        if ((y == 0) || ((Int)(cuPelY + y + roiHeight) >= picHeight)) //NOTE: RExt - is this still necessary?
        {
          continue;
        }

        Int tempY = y + relCUPelY + roiHeight - 1;

        for(Int x = max(srchRngHorLeft, -cuPelX); x <= srchRngHorRight; x++)
        {
          if ((x == 0) || ((Int)(cuPelX + x + roiWidth) >= picWidth)) //NOTE: RExt - is this still necessary?
          {
            continue;
          }

          Int tempX = x + relCUPelX + roiWidth - 1;

          if ((tempX >= 0) && (tempY >= 0))
          {
            Int iTempRasterIdx = (tempY/pcCU->getPic()->getMinCUHeight()) * pcCU->getPic()->getNumPartInCtuWidth() + (tempX/pcCU->getPic()->getMinCUWidth());
            Int iTempZscanIdx = g_auiRasterToZscan[iTempRasterIdx];
            if (iTempZscanIdx >= pcCU->getZorderIdxInCtu())
            {
              continue;
            }
          }

          if (!isValidIntraBCSearchArea(pcCU, x, y, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset))
          {
            continue;
          }

          sad = m_pcRdCost->getCostMultiplePreds( x, y);
          for(int r = 0; r < roiHeight; )
          {
            piRefSrch = piRefY + y * refStride + r*refStride + x;
            m_cDistParam.pCur = piRefSrch;
            m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

            sad += m_cDistParam.DistFunc( &m_cDistParam );
            if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
            {
              break;
            }

            r += 4;
          }

          xIntraBCSearchMVCandUpdate(sad, x, y, sadBestCand, cMVCand);
        }
      }

      bestX = cMVCand[0].getHor();
      bestY = cMVCand[0].getVer();
      sadBest = sadBestCand[0];
      if(sadBest - m_pcRdCost->getCostMultiplePreds( bestX, bestY) <= 16)
      {
        //chroma refine
        bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
        bestX       = cMVCand[bestCandIdx].getHor();
        bestY       = cMVCand[bestCandIdx].getVer();
        sadBest    = sadBestCand[bestCandIdx];
        rcMv.set( bestX, bestY );
        SAD       = sadBest;

        updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
        return;
      }

      for(Int y = (max(srchRngVerTop, -cuPelY) + 1); y <= srchRngVerBottom; y += 2)
      {
        if ((y == 0) || ((Int)(cuPelY + y + roiHeight) >= picHeight)) //NOTE: RExt - is this still necessary?
        {
          continue;
        }

        Int tempY = y + relCUPelY + roiHeight - 1;

        for(Int x = max(srchRngHorLeft, -cuPelX); x <= srchRngHorRight; x += 2)
        {
          if ((x == 0) || ((Int)(cuPelX + x + roiWidth) >= picWidth)) //NOTE: RExt - is this still necessary?
          {
            continue;
          }

          Int tempX = x + relCUPelX + roiWidth - 1;

          if ((tempX >= 0) && (tempY >= 0))
          {
            Int iTempRasterIdx = (tempY/pcCU->getPic()->getMinCUHeight()) * pcCU->getPic()->getNumPartInCtuWidth() + (tempX/pcCU->getPic()->getMinCUWidth());
            Int iTempZscanIdx = g_auiRasterToZscan[iTempRasterIdx];
            if (iTempZscanIdx >= pcCU->getZorderIdxInCtu())
            {
              continue;
            }
          }

          if (!isValidIntraBCSearchArea(pcCU, x, y, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset))
          {
            continue;
          }

          sad = m_pcRdCost->getCostMultiplePreds( x, y);
          for(int r = 0; r < roiHeight; )
          {
            piRefSrch = piRefY + y * refStride + r*refStride + x;
            m_cDistParam.pCur = piRefSrch;
            m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

            sad += m_cDistParam.DistFunc( &m_cDistParam );
            if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
            {
              break;
            }

            r += 4;
          }

          xIntraBCSearchMVCandUpdate(sad, x, y, sadBestCand, cMVCand);
          if(sadBestCand[0] <= 5)
          {
            //chroma refine & return
            bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
            bestX       = cMVCand[bestCandIdx].getHor();
            bestY       = cMVCand[bestCandIdx].getVer();
            sadBest    = sadBestCand[bestCandIdx];
            rcMv.set( bestX, bestY );
            SAD       = sadBest;

            updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
            return;
          }
        }
      }

      bestX = cMVCand[0].getHor();
      bestY = cMVCand[0].getVer();
      sadBest = sadBestCand[0];

      if((sadBest >= tempSadBest) || ((sadBest - m_pcRdCost->getCostMultiplePreds( bestX, bestY)) <= 32))
      {
        //chroma refine
        bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
        bestX       = cMVCand[bestCandIdx].getHor();
        bestY       = cMVCand[bestCandIdx].getVer();
        sadBest    = sadBestCand[bestCandIdx];
        rcMv.set( bestX, bestY );
        SAD       = sadBest;

        updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
        return;
      }

      tempSadBest = sadBestCand[0];


      for(Int y = (max(srchRngVerTop, -cuPelY) + 1); y <= srchRngVerBottom; y += 2)
      {
        if ((y == 0) || ((Int)(cuPelY + y + roiHeight) >= picHeight)) //NOTE: RExt - is this still necessary?
        {
          continue;
        }

        Int tempY = y + relCUPelY + roiHeight - 1;

        for(Int x = (max(srchRngHorLeft, -cuPelX) + 1); x <= srchRngHorRight; x += 2)
        {

          if ((x == 0) || ((Int)(cuPelX + x + roiWidth) >= picWidth)) //NOTE: RExt - is this still necessary?
          {
            continue;
          }

          Int tempX = x + relCUPelX + roiWidth - 1;

          if ((tempX >= 0) && (tempY >= 0))
          {
            Int iTempRasterIdx = (tempY/pcCU->getPic()->getMinCUHeight()) * pcCU->getPic()->getNumPartInCtuWidth() + (tempX/pcCU->getPic()->getMinCUWidth());
            Int iTempZscanIdx = g_auiRasterToZscan[iTempRasterIdx];
            if (iTempZscanIdx >= pcCU->getZorderIdxInCtu())
            {
              continue;
            }
          }

          if (!isValidIntraBCSearchArea(pcCU, x, y, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset))
          {
            continue;
          }

          sad = m_pcRdCost->getCostMultiplePreds( x, y);
          for(int r = 0; r < roiHeight; )
          {
            piRefSrch = piRefY + y * refStride + r*refStride + x;
            m_cDistParam.pCur = piRefSrch;
            m_cDistParam.pOrg = pcPatternKey->getROIY() + r * pcPatternKey->getPatternLStride();

            sad += m_cDistParam.DistFunc( &m_cDistParam );
            if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
            {
              break;
            }

            r += 4;
          }

          xIntraBCSearchMVCandUpdate(sad, x, y, sadBestCand, cMVCand);
          if(sadBestCand[0] <= 5)
          {
            //chroma refine & return
            bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
            bestX       = cMVCand[bestCandIdx].getHor();
            bestY       = cMVCand[bestCandIdx].getVer();
            sadBest    = sadBestCand[bestCandIdx];
            rcMv.set( bestX, bestY );
            SAD       = sadBest;

            updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
            return;
          }
        }
      }
    }
  }
  else //full search
  {
    setDistParamComp(COMPONENT_Y);
    piRefY += (srchRngVerBottom * refStride);
    Int picWidth = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
    Int picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();

    for(Int y = srchRngVerBottom; y >= srchRngVerTop; y--)
    {
      if ( ((Int)(cuPelY + y) < 0) || ((Int) (cuPelY + y + roiHeight) >= picHeight)) //NOTE: RExt - is this still necessary?
      {
        piRefY -= refStride;
        continue;
      }

      for(Int x = srchRngHorLeft; x <= srchRngHorRight; x++ )
      {

        if (((Int)(cuPelX + x) < 0) || ((Int) (cuPelX + x + roiWidth) >= picWidth)) //NOTE: RExt - is this still necessary?
        {
          continue;
        }

        Int tempX = x + relCUPelX + roiWidth - 1;
        Int tempY = y + relCUPelY + roiHeight - 1;
        if ((tempX >= 0) && (tempY >= 0))
        {
          Int iTempRasterIdx = (tempY/pcCU->getPic()->getMinCUHeight()) * pcCU->getPic()->getNumPartInCtuWidth() + (tempX/pcCU->getPic()->getMinCUWidth());
          Int iTempZscanIdx  = g_auiRasterToZscan[iTempRasterIdx];
          if (iTempZscanIdx >= pcCU->getZorderIdxInCtu())
          {
            continue;
          }
        }

        if (!isValidIntraBCSearchArea(pcCU, x, y, chromaROIWidthInPixels, chromaROIHeightInPixels, partOffset))
        {
          continue;
        }

        piRefSrch = piRefY + x;
        m_cDistParam.pCur = piRefSrch;

        m_cDistParam.bitDepth = pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA );
        sad = m_cDistParam.DistFunc( &m_cDistParam );

        sad += m_pcRdCost->getCostMultiplePreds( x, y);
        if ( sad < sadBest )
        {
          sadBest = sad;
          bestX    = x;
          bestY    = y;
        }
      }

      piRefY -= refStride;
    }
  }

  bestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, partOffset,partIdx);
  bestX       = cMVCand[bestCandIdx].getHor();
  bestY       = cMVCand[bestCandIdx].getVer();
  sadBest    = sadBestCand[bestCandIdx];
  rcMv.set( bestX, bestY );
  SAD       = sadBest;

  updateBVMergeCandLists(roiWidth, roiHeight, cMVCand);
}

Void TEncSearch::updateBVMergeCandLists(int roiWidth, int roiHeight, TComMv* mvCand)
{
  if(roiWidth+roiHeight > 8)
  {
    m_numBVs = MergeCandLists(m_acBVs, m_numBVs, mvCand, CHROMA_REFINEMENT_CANDIDATES, false);

    if(roiWidth+roiHeight==32)
    {
      m_numBV16s = m_numBVs;
    }
  }
}

Int TEncSearch::xIntraBCHashTableIndex(TComDataCU* pcCU, Int posX, Int posY, Int width, Int height, Bool isRec)
{
  TComPicYuv* HashPic;
  Pel*        plane;
  Int         numComp     = 1;
  UInt        hashIdx     = 0;
  UInt        grad        = 0;
  UInt        avgDC1      = 0;
  UInt        avgDC2      = 0;
  UInt        avgDC3      = 0;
  UInt        avgDC4      = 0;
  UInt        gradX       = 0;
  UInt        gradY       = 0;
  Int         picWidth    = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
  Int         picHeight   = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();
  Int         totalSamples = width * height;
  if(numComp == 3)
  {
    totalSamples = getTotalSamples(width, height,pcCU->getSlice()->getSPS()->getChromaFormatIdc());
  }

  assert((posX + width) <= picWidth);
  assert((posY + height) <= picHeight);

  if(isRec)
  {
    HashPic = pcCU->getPic()->getPicYuvRec();
  }
  else
  {
    if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && m_pcEncCfg->getRGBFormatFlag() )
    {
      HashPic = pcCU->getPic()->getPicYuvResi();
    }
    else
    {
      HashPic = pcCU->getPic()->getPicYuvOrg();
    }
  }

  for(Int chan=0; chan < numComp; chan++)
  {
    const ComponentID compID=ComponentID(chan);
    Int bitdepth   = pcCU->getSlice()->getSPS()->getBitDepth(toChannelType(compID));
    Int cxstride = HashPic->getStride(compID);
    Int cxposX = posX >> HashPic->getComponentScaleX(compID);
    Int cxposY = posY >> HashPic->getComponentScaleY(compID);
    Int cxwidth = width >> HashPic->getComponentScaleX(compID);
    Int cxheight = height >> HashPic->getComponentScaleY(compID);

    plane = HashPic->getAddr(compID) + cxposY * cxstride + cxposX;

    for (UInt y = 1; y < (cxheight >> 1); y++)
    {
      for (UInt x = 1; x < (cxwidth >> 1); x++)
      {
        avgDC1 += (plane[y*cxstride+x] >> (bitdepth - 8));
        gradX = abs(plane[y*cxstride+x] - plane[y*cxstride+x-1]) >> (bitdepth - 8);
        gradY = abs(plane[y*cxstride+x] - plane[(y-1)*cxstride+x]) >> (bitdepth - 8);
        grad += (gradX + gradY) >> 1;
      }
    }

    for (UInt y = (cxheight >> 1); y < cxheight; y++)
    {
      for (UInt x = 1; x < (cxwidth >> 1); x++)
      {
        avgDC2 += (plane[y*cxstride+x] >> (bitdepth - 8));
        gradX = abs(plane[y*cxstride+x] - plane[y*cxstride+x-1]) >> (bitdepth - 8);
        gradY = abs(plane[y*cxstride+x] - plane[(y-1)*cxstride+x]) >> (bitdepth - 8);
        grad += (gradX + gradY) >> 1;
      }
    }

    for (UInt y = 1; y < (cxheight >> 1); y++)
    {
      for (UInt x = (cxwidth >> 1); x < cxwidth; x++)
      {
        avgDC3 += plane[y*cxstride+x] >> (bitdepth - 8);
        gradX = abs(plane[y*cxstride+x] - plane[y*cxstride+x-1]) >> (bitdepth - 8);
        gradY = abs(plane[y*cxstride+x] - plane[(y-1)*cxstride+x]) >> (bitdepth - 8);
        grad += (gradX + gradY) >> 1;
      }
    }

    for (UInt y = (cxheight >> 1); y < cxheight; y++)
    {
      for (UInt x = (cxwidth >> 1); x < cxwidth; x++)
      {
        avgDC4 += plane[y*cxstride+x] >> (bitdepth - 8);
        gradX = abs(plane[y*cxstride+x] - plane[y*cxstride+x-1]) >> (bitdepth - 8);
        gradY = abs(plane[y*cxstride+x] - plane[(y-1)*cxstride+x]) >> (bitdepth - 8);
        grad += (gradX + gradY) >> 1;
      }
    }
  }

  avgDC1 = (avgDC1 << 2)/(totalSamples);
  avgDC2 = (avgDC2 << 2)/(totalSamples);
  avgDC3 = (avgDC3 << 2)/(totalSamples);
  avgDC4 = (avgDC4 << 2)/(totalSamples);

  grad = grad/(totalSamples);

  if(grad < 5 - (m_pcEncCfg->getRGBFormatFlag()))
  {
    return -1;
  }

  grad   = (grad >> 4) & 0xf; // 4 bits

  avgDC1 = (avgDC1>>5) & 0x7; // 3 bits
  avgDC2 = (avgDC2>>5) & 0x7;
  avgDC3 = (avgDC3>>5) & 0x7;
  avgDC4 = (avgDC4>>5) & 0x7;

  hashIdx = (avgDC1 << 13) + (avgDC2 << 10) + (avgDC3 << 7) + (avgDC4 << 4) + grad;

  assert(hashIdx <= 0XFFFF);

  return hashIdx;
}

Void TEncSearch::xIntraBCHashSearch( TComDataCU* pcCU, TComYuv* pcYuvOrg, Int partIdx, TComMv* pcMvPred, TComMv& rcMv, UInt intraBCECost)
{
  UInt      partAddr;
  Int       roiWidth;
  Int       roiHeight;

  TComYuv*    pcYuv = pcYuvOrg;

  TComPattern   cPattern;

  Int        orgHashIndex;

  Distortion  sadBestCand[CHROMA_REFINEMENT_CANDIDATES];
  TComMv      cMVCand[CHROMA_REFINEMENT_CANDIDATES];
  for(Int cand = 0; cand < CHROMA_REFINEMENT_CANDIDATES; cand++)
  {
    sadBestCand[cand] = std::numeric_limits<Distortion>::max();
    cMVCand[cand].set(0,0);
  }

  pcCU->getPartIndexAndSize( partIdx, partAddr, roiWidth, roiHeight );

#if MCTS_ENC_CHECK
  Int roiPosX, roiPosY;
  Int roiW, roiH;
  pcCU->getPartPosition(partIdx, roiPosX, roiPosY, roiW, roiH);
  assert(roiW == roiWidth);
  assert(roiH == roiHeight);
  cPattern.initPattern( pcYuv->getAddr(COMPONENT_Y, partAddr),
                        roiWidth,
                        roiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA),
                        roiPosX,
                        roiPosY);
  xInitTileBorders(pcCU, &cPattern);
#else
  cPattern.initPattern( pcYuv->getAddr  ( COMPONENT_Y, partAddr ),
                        roiWidth,
                        roiHeight,
                        pcYuv->getStride(COMPONENT_Y),
                        pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) );
#endif

  orgHashIndex = xIntraBCHashTableIndex(pcCU, pcCU->getCUPelX(), pcCU->getCUPelY(), roiWidth, roiHeight, false);

  if(orgHashIndex < 0)
  {
    return;
  }

  IntraBCHashNode* HashLinklist = getHashLinklist(0, orgHashIndex);  //Intra full frame hash search only for 8x8

  Pel*        piRefY     = pcCU->getPic()->getPicYuvRec()->getAddr( COMPONENT_Y);
  Int         refStride  = pcCU->getPic()->getPicYuvRec()->getStride(COMPONENT_Y);

  // disable weighted prediction
  setWpScalingDistParam( pcCU, -1, REF_PIC_LIST_X );

  m_pcRdCost->selectMotionLambda( true, 0, pcCU->getCUTransquantBypass(partAddr) );
  m_pcRdCost->setPredictor(*pcMvPred);
  m_pcRdCost->setCostScale  ( 0 );
  m_pcRdCost->setDistParam( &cPattern, piRefY, refStride,  m_cDistParam );

  setDistParamComp(COMPONENT_Y);
  m_cDistParam.bitDepth  = pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA );
  m_cDistParam.iRows     = 4;//to calculate the sad line by line;
  m_cDistParam.iSubShift = 0;

  Int  cuPelX     = pcCU->getCUPelX();
  Int  cuPelY     = pcCU->getCUPelY();
  const UInt maxCuWidth         = pcCU->getSlice()->getSPS()->getMaxCUWidth();
  const UInt maxCuHeight        = pcCU->getSlice()->getSPS()->getMaxCUHeight();
  const UInt searchWidth        = m_pcEncCfg->getUseIntraBCFullFrameSearch() ? 0 : pcCU->getIntraBCSearchAreaWidth( m_pcEncCfg->getIntraBCSearchWidthInCTUs() );
  const UInt nonHashSearchWidth = pcCU->getIntraBCSearchAreaWidth( m_pcEncCfg->getIntraBCNonHashSearchWidthInCTUs() );
  const UInt ctuPelX            = (cuPelX / maxCuWidth) * maxCuWidth;
  const UInt ctuPelY            = (cuPelY / maxCuHeight) * maxCuHeight;

  Distortion sad;

  Int         tempX;
  Int         tempY;
  Pel*  piRefSrch;

  xIntraBCSearchMVCandUpdate(intraBCECost, rcMv.getHor(), rcMv.getVer(), sadBestCand, cMVCand);

  const Int        relCUPelX    = cuPelX % maxCuWidth;
  const Int        relCUPelY    = cuPelY % maxCuHeight;
  const ChromaFormat format = pcCU->getPic()->getChromaFormat();
  const Int chromaROIWidthInPixels  = (((format == CHROMA_420) || (format == CHROMA_422)) && (roiWidth  == 4) && ((relCUPelX & 0x4) != 0)) ? (roiWidth  * 2) : roiWidth;
  const Int chromaROIHeightInPixels = (((format == CHROMA_420)                          ) && (roiHeight == 4) && ((relCUPelY & 0x4) != 0)) ? (roiHeight * 2) : roiHeight;

  while(HashLinklist)
  {
    tempX = HashLinklist->posX;
    tempY = HashLinklist->posY;

    if( !m_pcEncCfg->getUseIntraBCFullFrameSearch() )    // if full frame search is disabled, then apply following constraints on search range
    {
      const UInt refCuX    = tempX/maxCuWidth;
      const UInt refCuY    = tempY/maxCuHeight;
      const UInt refCuPelX = refCuX*maxCuWidth;
      const UInt refCuPelY = refCuY*maxCuHeight;
      if( refCuPelX+searchWidth < ctuPelX          // don't search left area of IntraBlockCopySearchWidth
        || refCuPelX+nonHashSearchWidth >= ctuPelX // don't search in the area that has been already searched in HashBasedIntraBlockCopySearch
        || refCuPelY != ctuPelY )                    // only search current CTU row
      {
        HashLinklist = HashLinklist->next;
        continue;
      }
    }

    Int xBv = tempX - cuPelX;
    Int yBv = tempY - cuPelY;
    if ( !isBlockVectorValid(cuPelX,cuPelY,roiWidth,roiHeight,pcCU,0,0,xBv,yBv,pcCU->getSlice()->getSPS()->getMaxCUWidth()))
    {
      HashLinklist = HashLinklist->next;
      continue;
    }
    Int xPred = tempX - cuPelX;
    Int yPred = tempY - cuPelY;

    Bool validCand = isValidIntraBCSearchArea(pcCU, xPred, yPred, chromaROIWidthInPixels, chromaROIHeightInPixels, partAddr);
    if( !validCand )
    {
      HashLinklist = HashLinklist->next;
      continue;
    }

    sad = 0;//m_pcRdCost->getCost( tempX - cuPelX, tempY - cuPelY);

    for(int r = 0; r < roiHeight; )
    {
      piRefSrch = piRefY + tempY * refStride + r*refStride + tempX;
      m_cDistParam.pCur = piRefSrch;
      m_cDistParam.pOrg = cPattern.getROIY() + r * cPattern.getPatternLStride();

      sad += m_cDistParam.DistFunc( &m_cDistParam );
      if (sad > sadBestCand[CHROMA_REFINEMENT_CANDIDATES - 1])
      {
        break;
      }

      r += 4;
    }

    if(sad <= 16)
    {
      sad += m_pcRdCost->getCostMultiplePreds( tempX - cuPelX, tempY - cuPelY);
      xIntraBCSearchMVCandUpdate(sad, tempX - cuPelX, tempY - cuPelY, sadBestCand, cMVCand);
      break;
    }

    sad += m_pcRdCost->getCostMultiplePreds( tempX - cuPelX, tempY - cuPelY);
    xIntraBCSearchMVCandUpdate(sad, tempX - cuPelX, tempY - cuPelY, sadBestCand, cMVCand);
    HashLinklist = HashLinklist->next;
  }

  Int  iBestCandIdx = xIntraBCSearchMVChromaRefine(pcCU, roiWidth, roiHeight, cuPelX, cuPelY, sadBestCand, cMVCand, 0,partIdx);
  rcMv = cMVCand[iBestCandIdx];

  m_numBVs = MergeCandLists(m_acBVs, m_numBVs, cMVCand, CHROMA_REFINEMENT_CANDIDATES, false);

  return;
}

Void TEncSearch::xIntraBCHashTableUpdate(TComDataCU* pcCU, Bool isRec)
{
  Int         roiWidth  = 8;
  Int         roiHeight = 8;
  Int         cuPelX    = pcCU->getCUPelX();
  Int         cuPelY    = pcCU->getCUPelY();
  Int         tempX;
  Int         tempY;
  Int         picWidth = pcCU->getSlice()->getSPS()->getPicWidthInLumaSamples();
  Int         picHeight = pcCU->getSlice()->getSPS()->getPicHeightInLumaSamples();
  UInt        maxCuWidth=pcCU->getSlice()->getSPS()->getMaxCUWidth();
  UInt        maxCuHeight=pcCU->getSlice()->getSPS()->getMaxCUHeight();
  Int         orgHashIndex;
  IntraBCHashNode* newHashNode;
  TComPicSym *pcSym = pcCU->getPic()->getPicSym();
  UInt       startCtu = !pcCU->getSlice()->getSliceMode() ? 0
                      : pcSym->getCtuTsToRsAddrMap(pcCU->getSlice()->getSliceSegmentCurStartCtuTsAddr());
  UInt       refY     = pcSym->getCtu(startCtu)->getCUPelY();

  const UInt lcuWidth = pcCU->getSlice()->getSPS()->getMaxCUWidth();
  const UInt lcuHeight = pcCU->getSlice()->getSPS()->getMaxCUHeight();

  const UInt curTileIdx = pcCU->getPic()->getPicSym()->getTileIdxMap( pcCU->getCtuRsAddr() );
  TComTile* curTile = pcCU->getPic()->getPicSym()->getTComTile( curTileIdx );

  const Int tileAreaRight  = (curTile->getRightEdgePosInCtus() + 1) * lcuWidth;
  const Int tileAreaBottom = (curTile->getBottomEdgePosInCtus() + 1) * lcuHeight;

  const Int tileAreaLeft   = tileAreaRight - curTile->getTileWidthInCtus() * lcuWidth;
  const Int tileAreaTop    = tileAreaBottom - curTile->getTileHeightInCtus() * lcuHeight;

  Int jStart = ( tileAreaTop  == cuPelY) ? 7 : 0;
  Int iStart = ( tileAreaLeft == cuPelX) ? 7 : 0;


  for(Int j = jStart; j < maxCuHeight; j++)
  {
    tempY = cuPelY - roiHeight + 1  + j;
    if ( pcCU->getSlice()->getSliceMode() && tempY < refY )
    {
      continue;
    }
    for(Int i = iStart; i < maxCuWidth; i++)
    {
      tempX = cuPelX - roiWidth + 1 + i;

      if((tempX < 0) || (tempY < 0) || ((tempX + roiWidth) >= picWidth) || ((tempY + roiHeight) >= picHeight))
      {
        continue;
      }
      if( pcCU->getSlice()->getSliceMode() )
      {
        Int      ctuX = tempX / pcCU->getSlice()->getSPS()->getMaxCUWidth();
        Int      ctuY = tempY / pcCU->getSlice()->getSPS()->getMaxCUHeight();
        UInt   refCtu = ctuX + pcSym->getFrameWidthInCtus()*ctuY;
        if ( refCtu < startCtu )
        {
          continue;
        }
      }

      orgHashIndex = xIntraBCHashTableIndex(pcCU, tempX, tempY, roiWidth, roiHeight, isRec);

      if(orgHashIndex < 0)
      {
        continue;
      }

      newHashNode = new IntraBCHashNode;

      assert(newHashNode);

      newHashNode->posX = tempX;
      newHashNode->posY = tempY;
      setHashLinklist(newHashNode, 0, orgHashIndex); //Intra full frame hash search only for 8x8
    }
  }
}

Void TEncSearch::xClearIntraBCHashTable()
{
  if(m_pcIntraBCHashTable)
  {
    for(int iDepth = 0; iDepth < INTRABC_HASH_DEPTH; iDepth++)
    {
      if(m_pcIntraBCHashTable[iDepth])
      {
        for(int idx = 0; idx < INTRABC_HASH_TABLESIZE; idx++)
        {
          if (m_pcIntraBCHashTable[iDepth][idx] == NULL)
          {
            continue;
          }
          else
          {
            while(m_pcIntraBCHashTable[iDepth][idx]->next)
            {
              IntraBCHashNode* tempNode = m_pcIntraBCHashTable[iDepth][idx]->next;
              m_pcIntraBCHashTable[iDepth][idx]->next = m_pcIntraBCHashTable[iDepth][idx]->next->next;

              delete tempNode;
            }

            delete m_pcIntraBCHashTable[iDepth][idx];
            m_pcIntraBCHashTable[iDepth][idx] = NULL;
          }
        }
      }
    }
  }
}

Void TEncSearch::setHashLinklist(IntraBCHashNode*& hashLinklist, UInt depth, UInt hashIdx)
{
  hashLinklist->next = m_pcIntraBCHashTable[depth][hashIdx];
  m_pcIntraBCHashTable[depth][hashIdx] = hashLinklist;
}

Void TEncSearch::xEstimateInterResidualQTTUCSC( TComYuv        *pcResi,
                                                Double         &rdCost,
                                                UInt           &bits,
                                                Distortion     &dist,
                                                TComTU         &rTu,
                                                TComYuv*       pcOrgResi,
                                                ACTRDTestTypes eACTRDType
                                                DEBUG_STRING_FN_DECLARE(sDebug) )
{
  TComDataCU *pcCU          = rTu.getCU();
  const UInt absPartIdx     = rTu.GetAbsPartIdxTU();
  const UInt depth          = rTu.GetTransformDepthTotal();
  const UInt trMode         = rTu.GetTransformDepthRel();
  const UInt partIdxesPerTU = rTu.GetAbsPartIdxNumParts();
  const UInt numValidComp   = pcCU->getPic()->getNumberValidComponents();
  const Bool extendedPrecision = pcCU->getSlice()->getSPS()->getSpsRangeExtension().getExtendedPrecisionProcessingFlag();

  assert( pcCU->getDepth( 0 ) == pcCU->getDepth( absPartIdx ) );
  const UInt uiLog2TrSize = rTu.GetLog2LumaTrSize();

  UInt splitFlag = ((pcCU->getSlice()->getSPS()->getQuadtreeTUMaxDepthInter() == 1) && pcCU->isInter(absPartIdx) && ( pcCU->getPartitionSize(absPartIdx) != SIZE_2Nx2N ));
  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(absPartIdx) && (pcCU->getWidth(absPartIdx) >= 32) && (pcCU->isInter(absPartIdx) || pcCU->isIntraBC(absPartIdx)) && (pcCU->getPartitionSize(absPartIdx) != SIZE_2Nx2N))
  {
    splitFlag = 1;
  }

  Bool bCheckFull;

  if ( splitFlag && depth == pcCU->getDepth(absPartIdx) && ( uiLog2TrSize >  pcCU->getQuadtreeTULog2MinSizeInCU(absPartIdx) ) )
  {
    bCheckFull = false;
  }
  else
  {
    bCheckFull =  ( uiLog2TrSize <= pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() );
  }

  Bool bCheckSplit  = ( uiLog2TrSize >  pcCU->getQuadtreeTULog2MinSizeInCU(absPartIdx) );
  if(m_pcEncCfg->getTransquantBypassInferTUSplit() && pcCU->isLosslessCoded(absPartIdx) && (pcCU->isIntraBC(absPartIdx) || pcCU->isInter(absPartIdx)) && (pcCU->getWidth(absPartIdx) >= 32) && bCheckFull)
  {
    bCheckSplit = false;
  }

  assert( bCheckFull || bCheckSplit );
  assert( pcCU->getPic()->getChromaFormat() == CHROMA_444 );

  UInt       singleColorSpaceId      = 0;
  Double     dSingleCost             = MAX_DOUBLE;
  Distortion singleDist              = 0;
  UInt       singleBits              = 0;

  Double     dSingleColorSpaceCost[2] = {MAX_DOUBLE, MAX_DOUBLE};
  Distortion singleColorSpaceDist[2]  = {0, 0};
  UInt       singleColorSpaceBits[2]  = {0, 0};

  Distortion singleDistComp[2][MAX_NUM_COMPONENT]               = { {0,0,0}, {0,0,0} };
  TCoeff     absSum[2][MAX_NUM_COMPONENT]                       = { {0,0,0}, {0,0,0} };
  UInt       bestTransformMode[2][MAX_NUM_COMPONENT]            = { {0,0,0}, {0,0,0} };
  UInt       bestExplicitRdpcmModeUnSplit[2][MAX_NUM_COMPONENT] = { {3,3,3}, {3,3,3} };
  SChar      bestCrossCPredictionAlpha[2][MAX_NUM_COMPONENT]    = { {0,0,0}, {0,0,0} };

  m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );

  if( bCheckFull )
  {
    Double minCost[MAX_NUM_COMPONENT];
    Bool checkTransformSkip[MAX_NUM_COMPONENT];
    pcCU->setTrIdxSubParts( trMode, absPartIdx, depth );

    m_pcEntropyCoder->resetBits();

    memset( m_pTempPel, 0, sizeof( Pel ) * rTu.getRect(COMPONENT_Y).width * rTu.getRect(COMPONENT_Y).height ); // not necessary needed for inside of recursion (only at the beginning)

    const UInt QTTempAccessLayer = pcCU->getSlice()->getSPS()->getQuadtreeTULog2MaxSize() - uiLog2TrSize;
    TCoeff *pcCoeffCurr[MAX_NUM_COMPONENT];
#if ADAPTIVE_QP_SELECTION
    TCoeff *pcArlCoeffCurr[MAX_NUM_COMPONENT];
#endif

    TCoeff bestColorSpaceCoeff[MAX_NUM_COMPONENT][MAX_TU_SIZE*MAX_TU_SIZE];
    Pel    bestColorSpaceResi [MAX_NUM_COMPONENT][MAX_TU_SIZE*MAX_TU_SIZE];
#if ADAPTIVE_QP_SELECTION
    TCoeff bestColorSpaceArlCoeff[MAX_NUM_COMPONENT][MAX_TU_SIZE*MAX_TU_SIZE];
#endif

    for(Int colorSpaceId = 0; colorSpaceId < 2; colorSpaceId++)
    {
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
      m_pcEntropyCoder->resetBits();

      const Bool iColorTransform = m_pcEncCfg->getRGBFormatFlag()? (!colorSpaceId? true: false): (colorSpaceId? true: false);

      if(eACTRDType == ACT_TRAN_CLR && !iColorTransform)
      {
        continue;
      }
      if(eACTRDType == ACT_ORG_CLR && iColorTransform)
      {
        continue;
      }

      pcCU->setColourTransformSubParts(iColorTransform, absPartIdx, depth);
      for( UInt ch = 0; ch < numValidComp; ch++ )
      {
        const ComponentID compID = ComponentID(ch);
        if(iColorTransform)
        {
          m_pcNoCorrYuvTmp[pcCU->getDepth(absPartIdx)].copyPartToPartComponent( compID, pcResi, absPartIdx, rTu.getRect(compID).width, rTu.getRect(compID).height );
        }
        else
        {
          pcOrgResi->copyPartToPartComponent( compID, pcResi, absPartIdx, rTu.getRect(compID).width, rTu.getRect(compID).height );
        }
      }

      for(UInt i = 0; i < numValidComp; i++)
      {
        minCost[i] = MAX_DOUBLE;
      }

      Pel crossCPredictedResidualBuffer[ MAX_TU_SIZE * MAX_TU_SIZE ];

      for(UInt i = 0; i < numValidComp; i++)
      {
        checkTransformSkip[i]    = false;
        const ComponentID compID = ComponentID(i);
        const Int channelBitDepth=pcCU->getSlice()->getSPS()->getBitDepth(toChannelType(compID));
        pcCoeffCurr[compID]      = m_ppcQTTempCoeff[compID][QTTempAccessLayer] + rTu.getCoefficientOffset(compID);
#if ADAPTIVE_QP_SELECTION
        pcArlCoeffCurr[compID] = m_ppcQTTempArlCoeff[compID ][QTTempAccessLayer] +  rTu.getCoefficientOffset(compID);
#endif

        if(rTu.ProcessComponentSection(compID))
        {
          QpParam cQP(*pcCU, compID, absPartIdx);
          if(!pcCU->isLosslessCoded(0) && iColorTransform)
          {
            Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
            m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( deltaQP );
            m_pcRdCost->adjustLambdaForColourTrans( deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
          }

          checkTransformSkip[compID] = pcCU->getSlice()->getPPS()->getUseTransformSkip() &&
                                       TUCompRectHasAssociatedTransformSkipFlag(rTu.getRect(compID), pcCU->getSlice()->getPPS()->getPpsRangeExtension().getLog2MaxTransformSkipBlockSize()) &&
                                       (!pcCU->isLosslessCoded(0));

          assert(rTu.getRect(compID).width == rTu.getRect(compID).height);
          const TComRectangle &tuCompRect       = rTu.getRect(compID);
          TCoeff        *currentCoefficients    = pcCoeffCurr[compID];
#if ADAPTIVE_QP_SELECTION
          TCoeff        *currentARLCoefficients = pcArlCoeffCurr[compID];
#endif
          const Bool isCrossCPredictionAvailable =    isChroma(compID)
                                                   && pcCU->getSlice()->getPPS()->getPpsRangeExtension().getCrossComponentPredictionEnabledFlag()
                                                   && (pcCU->getCbf(absPartIdx, COMPONENT_Y, trMode) != 0);

          SChar preCalcAlpha = 0;
          const Pel *pLumaResi = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( COMPONENT_Y, rTu.getRect( COMPONENT_Y ).x0, rTu.getRect( COMPONENT_Y ).y0 );

          if (isCrossCPredictionAvailable)
          {
            const Bool bUseReconstructedResidualForEstimate = m_pcEncCfg->getUseReconBasedCrossCPredictionEstimate();
            const Pel  *const lumaResidualForEstimate       = bUseReconstructedResidualForEstimate ? pLumaResi                                                     : pcResi->getAddrPix(COMPONENT_Y, tuCompRect.x0, tuCompRect.y0);
            const UInt        lumaResidualStrideForEstimate = bUseReconstructedResidualForEstimate ? m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y) : pcResi->getStride(COMPONENT_Y);

            preCalcAlpha = xCalcCrossComponentPredictionAlpha(rTu,
                                                              compID,
                                                              lumaResidualForEstimate,
                                                              pcResi->getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                              tuCompRect.width,
                                                              tuCompRect.height,
                                                              lumaResidualStrideForEstimate,
                                                              pcResi->getStride(compID));
          }

          const Int transformSkipModesToTest    = checkTransformSkip[compID] ? 2 : 1;
          const Int crossCPredictionModesToTest = (preCalcAlpha != 0)        ? 2 : 1; //preCalcAlpha cannot be anything other than 0 if isCrossCPredictionAvailable is false

          const Bool isOneMode                  = (crossCPredictionModesToTest == 1) && (transformSkipModesToTest == 1);

          for (Int transformSkipModeId = 0; transformSkipModeId < transformSkipModesToTest; transformSkipModeId++)
          {
            pcCU->setTransformSkipPartRange(transformSkipModeId, compID, absPartIdx, partIdxesPerTU);

            for (Int crossCPredictionModeId = 0; crossCPredictionModeId < crossCPredictionModesToTest; crossCPredictionModeId++)
            {
              const Bool isFirstMode          = (transformSkipModeId == 0) && (crossCPredictionModeId == 0);
              const Bool bUseCrossCPrediction = crossCPredictionModeId != 0;

              m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
              m_pcEntropyCoder->resetBits();

              pcCU->setTransformSkipPartRange(transformSkipModeId, compID, absPartIdx, partIdxesPerTU);
              pcCU->setCrossComponentPredictionAlphaPartRange((bUseCrossCPrediction ? preCalcAlpha : 0), compID, absPartIdx, partIdxesPerTU );

              if ((compID != COMPONENT_Cr) && ((transformSkipModeId == 1) ? m_pcEncCfg->getUseRDOQTS() : m_pcEncCfg->getUseRDOQ()))
              {
                COEFF_SCAN_TYPE scanType = COEFF_SCAN_TYPE(pcCU->getCoefScanIdx(absPartIdx, tuCompRect.width, tuCompRect.height, compID));
                m_pcEntropyCoder->estimateBit(m_pcTrQuant->m_pcEstBitsSbac, tuCompRect.width, tuCompRect.height, toChannelType(compID), scanType);
              }

#if RDOQ_CHROMA_LAMBDA
              m_pcTrQuant->selectLambda(compID);
#endif

              Pel *pcResiCurrComp = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
              UInt resiStride     = m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID);

              TCoeff bestCoeffComp   [MAX_TU_SIZE*MAX_TU_SIZE];
              Pel    bestResiComp    [MAX_TU_SIZE*MAX_TU_SIZE];

#if ADAPTIVE_QP_SELECTION
              TCoeff bestArlCoeffComp[MAX_TU_SIZE*MAX_TU_SIZE];
#endif
              TCoeff     currAbsSum   = 0;
              UInt       currCompBits = 0;
              Distortion currCompDist = 0;
              Double     currCompCost = 0;
              UInt       nonCoeffBits = 0;
              Distortion nonCoeffDist = 0;
              Double     nonCoeffCost = 0;

              if(!isOneMode && !isFirstMode)
              {
                memcpy(bestCoeffComp,    currentCoefficients,    (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
                memcpy(bestArlCoeffComp, currentARLCoefficients, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif
                for(Int y = 0; y < tuCompRect.height; y++)
                {
                  memcpy(&bestResiComp[y * tuCompRect.width], (pcResiCurrComp + (y * resiStride)), (sizeof(Pel) * tuCompRect.width));
                }
              }

              if (bUseCrossCPrediction)
              {
                TComTrQuant::crossComponentPrediction(rTu,
                                                      compID,
                                                      pLumaResi,
                                                      pcResi->getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                      crossCPredictedResidualBuffer,
                                                      tuCompRect.width,
                                                      tuCompRect.height,
                                                      m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y),
                                                      pcResi->getStride(compID),
                                                      tuCompRect.width,
                                                      false);

                m_pcTrQuant->transformNxN(rTu, compID, crossCPredictedResidualBuffer, tuCompRect.width, currentCoefficients,
#if ADAPTIVE_QP_SELECTION
                                          currentARLCoefficients,
#endif
                                          currAbsSum, cQP);
              }
              else
              {
                m_pcTrQuant->transformNxN(rTu, compID, pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ), pcResi->getStride(compID), currentCoefficients,
#if ADAPTIVE_QP_SELECTION
                                          currentARLCoefficients,
#endif
                                          currAbsSum, cQP);
              }

              if(isFirstMode || (currAbsSum == 0))
              {
                if (bUseCrossCPrediction)
                {
                  TComTrQuant::crossComponentPrediction(rTu,
                                                        compID,
                                                        pLumaResi,
                                                        m_pTempPel,
                                                        m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                        tuCompRect.width,
                                                        tuCompRect.height,
                                                        m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y),
                                                        tuCompRect.width,
                                                        m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID),
                                                        true);

                  nonCoeffDist = m_pcRdCost->getDistPart( channelBitDepth, m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getStride( compID ), pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          pcResi->getStride(compID), tuCompRect.width, tuCompRect.height, compID); // initialized with zero residual distortion
                }
                else
                {
                  nonCoeffDist = m_pcRdCost->getDistPart( channelBitDepth, m_pTempPel, tuCompRect.width, pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          pcResi->getStride(compID), tuCompRect.width, tuCompRect.height, compID); // initialized with zero residual distortion
                }

                m_pcEntropyCoder->encodeQtCbfZero( rTu, toChannelType(compID) );

                if ( isCrossCPredictionAvailable )
                {
                  m_pcEntropyCoder->encodeCrossComponentPrediction( rTu, compID );
                }

                nonCoeffBits = m_pcEntropyCoder->getNumberOfWrittenBits();
                nonCoeffCost = m_pcRdCost->calcRdCost( nonCoeffBits, nonCoeffDist );
              }

                if( currAbsSum > 0 ) //if non-zero coefficients are present, a residual needs to be derived for further prediction
                {
                  if (isFirstMode)
                  {
                    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
                    m_pcEntropyCoder->resetBits();
                  }

                  m_pcEntropyCoder->encodeQtCbf( rTu, compID, true );

                  if (isCrossCPredictionAvailable)
                  {
                    m_pcEntropyCoder->encodeCrossComponentPrediction( rTu, compID );
                  }

                  m_pcEntropyCoder->encodeCoeffNxN( rTu, currentCoefficients, compID );
                  currCompBits = m_pcEntropyCoder->getNumberOfWrittenBits();

                  pcResiCurrComp = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 );

                  m_pcTrQuant->invTransformNxN( rTu, compID, pcResiCurrComp, m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID), currentCoefficients, cQP DEBUG_STRING_PASS_INTO_OPTIONAL(&sSingleStringTest, (DebugOptionList::DebugString_InvTran.getInt()&debugPredModeMask)) );

                  if (bUseCrossCPrediction)
                  {
                    TComTrQuant::crossComponentPrediction(rTu,
                                                          compID,
                                                          pLumaResi,
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                          tuCompRect.width,
                                                          tuCompRect.height,
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y),
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID     ),
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID     ),
                                                          true);
                  }

                  currCompDist = m_pcRdCost->getDistPart( channelBitDepth, m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID),
                                                          pcResi->getAddrPix( compID, tuCompRect.x0, tuCompRect.y0 ),
                                                          pcResi->getStride(compID),
                                                          tuCompRect.width, tuCompRect.height, compID);

                  currCompCost = m_pcRdCost->calcRdCost(currCompBits, currCompDist);

                  if (pcCU->isLosslessCoded(0))
                  {
                    nonCoeffCost = MAX_DOUBLE;
                  }
                }
                else if ((transformSkipModeId == 1) && !bUseCrossCPrediction) //NOTE: RExt - if the CBF (i.e. currAbsSum) is 0, this mode combination gives the same result as when transformSkipModeId = 0 (Not coding-efficiency-affecting - maybe remove test in later revision)
                {
                  currCompCost = MAX_DOUBLE;
                }
                else
                {
                  currCompBits = nonCoeffBits;
                  currCompDist = nonCoeffDist;
                  currCompCost = nonCoeffCost;
                }

                // evaluate
                if ((currCompCost < minCost[compID]) || ((transformSkipModeId == 1) && (currCompCost == minCost[compID])))
                {
                  bestExplicitRdpcmModeUnSplit[colorSpaceId][compID] = pcCU->getExplicitRdpcmMode(compID, absPartIdx);

                  if(isFirstMode) //check for forced null
                  {
                    if((nonCoeffCost < currCompCost) || (currAbsSum == 0))
                    {
                      memset(currentCoefficients, 0, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));

                      currAbsSum   = 0;
                      currCompBits = nonCoeffBits;
                      currCompDist = nonCoeffDist;
                      currCompCost = nonCoeffCost;
                    }
                  }

                  absSum                 [colorSpaceId][compID] = currAbsSum;
                  singleDistComp         [colorSpaceId][compID] = currCompDist;
                  minCost                  [compID]               = currCompCost;
                  bestTransformMode      [colorSpaceId][compID] = transformSkipModeId;
                  bestCrossCPredictionAlpha[colorSpaceId][compID] = (crossCPredictionModeId == 1) ? pcCU->getCrossComponentPredictionAlpha(absPartIdx, compID) : 0;

                  if (absSum[colorSpaceId][compID] == 0)
                  {
                    if (bUseCrossCPrediction)
                    {
                      TComTrQuant::crossComponentPrediction(rTu,
                                                            compID,
                                                            pLumaResi,
                                                            m_pTempPel,
                                                            m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0),
                                                            tuCompRect.width,
                                                            tuCompRect.height,
                                                            m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y),
                                                            tuCompRect.width,
                                                            m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID),
                                                            true);
                    }
                    else
                    {
                      pcResiCurrComp = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
                      const UInt uiStride = m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID);
                      for(UInt uiY = 0; uiY < tuCompRect.height; uiY++)
                      {
                        memset(pcResiCurrComp, 0, (sizeof(Pel) * tuCompRect.width));
                        pcResiCurrComp += uiStride;
                      }
                    }
                  }
                }
                else
                {
                  // reset
                  memcpy(currentCoefficients,    bestCoeffComp,    (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
                  memcpy(currentARLCoefficients, bestArlCoeffComp, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif
                  for (Int y = 0; y < tuCompRect.height; y++)
                  {
                    memcpy((pcResiCurrComp + (y * resiStride)), &bestResiComp[y * tuCompRect.width], (sizeof(Pel) * tuCompRect.width));
                  }
                }
            }
          }

          pcCU->setExplicitRdpcmModePartRange            (   bestExplicitRdpcmModeUnSplit[colorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU);
          pcCU->setTransformSkipPartRange                (   bestTransformMode         [colorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU );
          pcCU->setCbfPartRange                          ((((absSum                    [colorSpaceId][compID] > 0) ? 1 : 0) << trMode), compID, absPartIdx, partIdxesPerTU );
          pcCU->setCrossComponentPredictionAlphaPartRange(   bestCrossCPredictionAlpha   [colorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU );


          if(!pcCU->isLosslessCoded(0) && iColorTransform)
          {
            Int deltaQP = pcCU->getSlice()->getPPS()->getPpsScreenExtension().getActQpOffset(compID) + pcCU->getSlice()->getSliceActQpDelta(compID);
            m_pcTrQuant->adjustBitDepthandLambdaForColourTrans( - deltaQP );
            m_pcRdCost->adjustLambdaForColourTrans( - deltaQP, pcCU->getSlice()->getSPS()->getBitDepths() );
          }
        } // processing section
      } // component loop

      if(iColorTransform)
      {
        m_pcQTTempTComYuv[QTTempAccessLayer].convert( extendedPrecision, rTu.getRect(COMPONENT_Y).x0, rTu.getRect(COMPONENT_Y).y0, rTu.getRect(COMPONENT_Y).width, false, pcCU->getSlice()->getSPS()->getBitDepths(), pcCU->isLosslessCoded(absPartIdx) );  //YcgCo -> RGB

        const TComRectangle &tuCompRect = rTu.getRect(COMPONENT_Y);
        singleDistComp[colorSpaceId][COMPONENT_Y ] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA ), m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( COMPONENT_Y, tuCompRect.x0, tuCompRect.y0 ), m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Y),
                                                                               pcOrgResi->getAddrPix( COMPONENT_Y, tuCompRect.x0, tuCompRect.y0 ), pcOrgResi->getStride(COMPONENT_Y), tuCompRect.width, tuCompRect.height, COMPONENT_Y );

        const TComRectangle &tuCompRectC = rTu.getRect(COMPONENT_Cb);
        singleDistComp[colorSpaceId][COMPONENT_Cb] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( COMPONENT_Cb, tuCompRectC.x0, tuCompRectC.y0 ), m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Cb),
                                                                               pcOrgResi->getAddrPix( COMPONENT_Cb, tuCompRectC.x0, tuCompRectC.y0 ), pcOrgResi->getStride(COMPONENT_Cb), tuCompRectC.width, tuCompRectC.height, COMPONENT_Cb );
        singleDistComp[colorSpaceId][COMPONENT_Cr] = m_pcRdCost->getDistPart(pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ), m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix( COMPONENT_Cr, tuCompRectC.x0, tuCompRectC.y0 ), m_pcQTTempTComYuv[QTTempAccessLayer].getStride(COMPONENT_Cr),
                                                                               pcOrgResi->getAddrPix( COMPONENT_Cr, tuCompRectC.x0, tuCompRectC.y0 ), pcOrgResi->getStride(COMPONENT_Cr), tuCompRectC.width, tuCompRectC.height, COMPONENT_Cr );
      }

      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
      m_pcEntropyCoder->resetBits();

      if( uiLog2TrSize > pcCU->getQuadtreeTULog2MinSizeInCU(absPartIdx) )
      {
        m_pcEntropyCoder->encodeTransformSubdivFlag( 0, 5 - uiLog2TrSize );
      }

      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        const UInt chOrderChange = ((ch + 1) == numValidComp) ? 0 : (ch + 1);
        const ComponentID compID=ComponentID(chOrderChange);
        if( rTu.ProcessComponentSection(compID) )
        {
          m_pcEntropyCoder->encodeQtCbf( rTu, compID, true );
        }
      }

      if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && (pcCU->getCbf(absPartIdx, COMPONENT_Y, trMode) || pcCU->getCbf(absPartIdx, COMPONENT_Cb, trMode) || pcCU->getCbf(absPartIdx, COMPONENT_Cr, trMode)) )
      {
        m_pcEntropyCoder->m_pcEntropyCoderIf->codeColourTransformFlag( pcCU, absPartIdx );
      }

      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        const ComponentID compID = ComponentID(ch);
        if (rTu.ProcessComponentSection(compID))
        {
          if(isChroma(compID) && (absSum[colorSpaceId][COMPONENT_Y] != 0))
          {
            m_pcEntropyCoder->encodeCrossComponentPrediction( rTu, compID );
          }

          m_pcEntropyCoder->encodeCoeffNxN( rTu, pcCoeffCurr[compID], compID );

          singleColorSpaceDist[colorSpaceId] += singleDistComp[colorSpaceId][compID];
        }
      }

      singleColorSpaceBits[colorSpaceId] = m_pcEntropyCoder->getNumberOfWrittenBits();
      dSingleColorSpaceCost[colorSpaceId]  = m_pcRdCost->calcRdCost( singleColorSpaceBits[colorSpaceId], singleColorSpaceDist[colorSpaceId] );

      if( dSingleColorSpaceCost[colorSpaceId] < dSingleCost )
      {
        dSingleCost          = dSingleColorSpaceCost[colorSpaceId];
        singleDist         = singleColorSpaceDist[colorSpaceId];
        singleBits         = singleColorSpaceBits[colorSpaceId];
        singleColorSpaceId = colorSpaceId;

        for(UInt i = 0; i < numValidComp; i++)
        {
          const ComponentID compID        = ComponentID(i);
          const TComRectangle &tuCompRect = rTu.getRect(compID);

          TCoeff *pcCoeff                 = m_ppcQTTempCoeff[compID][QTTempAccessLayer] + rTu.getCoefficientOffset(compID);
          memcpy(bestColorSpaceCoeff[i],    pcCoeff,    (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
          TCoeff *pcArlCoeff              = m_ppcQTTempArlCoeff[compID ][QTTempAccessLayer] + rTu.getCoefficientOffset(compID);
          memcpy(bestColorSpaceArlCoeff[i], pcArlCoeff, (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif

          Pel *piResi = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
          UInt resiStride = m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID);
          for(Int y = 0; y < tuCompRect.height; y++)
          {
            memcpy(&(bestColorSpaceResi[i][y * tuCompRect.width]), (piResi + (y * resiStride)), (sizeof(Pel) * tuCompRect.width));
          }
        }
        m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ depth ][ CI_CHROMA_INTRA ] );  //CI_CHROMA_INTRA is never used
      }

      if( !colorSpaceId && !absSum[colorSpaceId][COMPONENT_Y] && !absSum[colorSpaceId][COMPONENT_Cb] && !absSum[colorSpaceId][COMPONENT_Cr] )  //all cbfs are zero, skip the other color space
        break;
    }

    Bool iColorTransform = m_pcEncCfg->getRGBFormatFlag()? (!singleColorSpaceId? true: false): (singleColorSpaceId? true: false);
    pcCU->setColourTransformSubParts(iColorTransform, absPartIdx, depth);

    for( UInt ch = 0; ch < numValidComp; ch++ )
    {
      ComponentID compID = ComponentID(ch);
      const TComRectangle &tuCompRect = rTu.getRect(compID);

      pcCU->setExplicitRdpcmModePartRange            (   bestExplicitRdpcmModeUnSplit[singleColorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU );
      pcCU->setTransformSkipPartRange                (   bestTransformMode         [singleColorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU );
      pcCU->setCbfPartRange                          ((((absSum                    [singleColorSpaceId][compID] > 0) ? 1 : 0) << trMode), compID, absPartIdx, partIdxesPerTU );
      pcCU->setCrossComponentPredictionAlphaPartRange(   bestCrossCPredictionAlpha   [singleColorSpaceId][compID],                            compID, absPartIdx, partIdxesPerTU );

      TCoeff *pcCoeff                 = m_ppcQTTempCoeff[compID][QTTempAccessLayer] + rTu.getCoefficientOffset(compID);
      memcpy( pcCoeff, bestColorSpaceCoeff[ch],   (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#if ADAPTIVE_QP_SELECTION
      TCoeff *pcArlCoeff              = m_ppcQTTempArlCoeff[compID ][QTTempAccessLayer] + rTu.getCoefficientOffset(compID);
      memcpy( pcArlCoeff, bestColorSpaceArlCoeff[ch], (sizeof(TCoeff) * tuCompRect.width * tuCompRect.height));
#endif

      Pel *piResi = m_pcQTTempTComYuv[QTTempAccessLayer].getAddrPix(compID, tuCompRect.x0, tuCompRect.y0);
      UInt resiStride = m_pcQTTempTComYuv[QTTempAccessLayer].getStride(compID);
      for(Int y = 0; y < tuCompRect.height; y++)
      {
        memcpy((piResi + (y * resiStride)), &(bestColorSpaceResi[ch][y * tuCompRect.width]), (sizeof(Pel) * tuCompRect.width));
      }
    }

    m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_CHROMA_INTRA ] );
  } // check full

  // code sub-blocks
  if( bCheckSplit )
  {
    if( bCheckFull )
    {
      m_pcRDGoOnSbacCoder->store( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_TEST ] );
      m_pcRDGoOnSbacCoder->load ( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
    }
    Distortion uiSubdivDist = 0;
    UInt       uiSubdivBits = 0;
    Double     dSubdivCost = 0.0;

    //save the non-split CBFs in case we need to restore them later

    UInt bestCBF     [MAX_NUM_COMPONENT];
    for(UInt ch = 0; ch < numValidComp; ch++)
    {
      const ComponentID compID=ComponentID(ch);

      if (rTu.ProcessComponentSection(compID))
      {
        bestCBF[compID] = pcCU->getCbf(absPartIdx, compID, trMode);
      }
    }

    TComTURecurse tuRecurseChild(rTu, false);
    const UInt QPartNumSubdiv = tuRecurseChild.GetAbsPartIdxNumParts();

      do
      {
        xEstimateInterResidualQTTUCSC( pcResi, dSubdivCost, uiSubdivBits, uiSubdivDist, tuRecurseChild, pcOrgResi, eACTRDType DEBUG_STRING_PASS_INTO(sSplitString));
      }
      while ( tuRecurseChild.nextSection(rTu) ) ;

      UInt cbfAny =0;
      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        UInt YUVCbf = 0;
        for( UInt ui = 0; ui < 4; ++ui )
        {
          YUVCbf |= pcCU->getCbf( absPartIdx + ui * QPartNumSubdiv, ComponentID(ch),  trMode + 1 );
        }
        UChar *pBase=pcCU->getCbf( ComponentID(ch) );
        const UInt flags = YUVCbf << trMode;
        for( UInt ui = 0; ui < 4 * QPartNumSubdiv; ++ui )
        {
          pBase[absPartIdx + ui] |= flags;
        }
        cbfAny |= YUVCbf;
      }

      m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_ROOT ] );
      m_pcEntropyCoder->resetBits();

      // when compID isn't a channel, code Cbfs:
      xEncodeInterResidualQT( MAX_NUM_COMPONENT, rTu );
      for(UInt ch = 0; ch < numValidComp; ch++)
      {
        xEncodeInterResidualQT( ComponentID(ch), rTu );
      }

      uiSubdivBits = m_pcEntropyCoder->getNumberOfWrittenBits();
      dSubdivCost  = m_pcRdCost->calcRdCost( uiSubdivBits, uiSubdivDist );

      if (!bCheckFull || (cbfAny && (dSubdivCost < dSingleCost)))
      {
        rdCost += dSubdivCost;
        bits += uiSubdivBits;
        dist += uiSubdivDist;
      }
      else
      {
        rdCost  += dSingleCost;
        bits += singleBits;
        dist += singleDist;

        //restore state to unsplit

        pcCU->setTrIdxSubParts( trMode, absPartIdx, depth );

        Bool iColorTransform = m_pcEncCfg->getRGBFormatFlag()? (!singleColorSpaceId? true: false): (singleColorSpaceId? true: false);
        pcCU->setColourTransformSubParts(iColorTransform, absPartIdx, depth);

        for(UInt ch = 0; ch < numValidComp; ch++)
        {
          const ComponentID compID = ComponentID(ch);
            if (rTu.ProcessComponentSection(compID))
            {
              pcCU->setCbfPartRange((bestCBF[compID] << trMode), compID, absPartIdx, partIdxesPerTU);
              pcCU->setCrossComponentPredictionAlphaPartRange(bestCrossCPredictionAlpha[singleColorSpaceId][compID], compID, absPartIdx, partIdxesPerTU);
              pcCU->setTransformSkipPartRange(bestTransformMode[singleColorSpaceId][compID], compID, absPartIdx, partIdxesPerTU);
              pcCU->setExplicitRdpcmModePartRange(bestExplicitRdpcmModeUnSplit[singleColorSpaceId][compID], compID, absPartIdx, partIdxesPerTU);
            }
        }

        m_pcRDGoOnSbacCoder->load( m_pppcRDSbacCoder[ depth ][ CI_QT_TRAFO_TEST ] );
      }
  }
  else
  {
    rdCost  += dSingleCost;
    bits += singleBits;
    dist += singleDist;
  }
}

Void TEncSearch::xPreCalcPaletteIndexRD(TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt paletteSize, TComRdCost *pcCost, UInt calcErroBits)
{
  Bool bLossless = pcCU->getCUTransquantBypass(0);
  Int errorLimit = bLossless ? 0 : 3 * m_paletteErrLimit * m_paletteErrLimit;

  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

  UInt paletteIdx, minError, bestIdx, pos;
  UChar useEscapeFlag=0;
  Int maxSpsPaletteSize = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();

  Pel distAdjY = DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA] - 8) << 1);
  Pel distAdjC = DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1);

  Short temp;
  UInt absError;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      UInt* indError = m_indError[pos];
      Int localAdjC = distAdjC;
      Bool discardChroma = y&scaleY || x&scaleX || pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400;
      if (discardChroma) localAdjC+=SCM_V0034_PALETTE_CHROMA_SHIFT_ADJ;
      UInt round = localAdjC ? (1 << (localAdjC-1)) : 0;
      bestIdx=0;
      minError = MAX_UINT;

      paletteIdx = 0;
      while (paletteIdx < paletteSize)
      {
        if( bLossless )
        {
          if( Palette[0][paletteIdx] != pSrc[0][pos] )
          {
            indError[paletteIdx] = absError = MAX_UINT;
          }
          else if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            absError  = abs(Palette[1][paletteIdx] - pSrc[1][posC]);
            absError += abs(Palette[2][paletteIdx] - pSrc[2][posC]);
            if (absError && !discardChroma)
            {
              indError[paletteIdx] = absError = MAX_UINT;
            }
            else
            {
              absError = std::min(1U, absError);
              indError[paletteIdx] = 0;
            }
          }
          else
          {
            absError = 0;
            indError[paletteIdx] = 0;
          }
        }
        else
        {
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            temp = Palette[1][paletteIdx] - pSrc[1][posC];
            absError = temp * temp;
            temp = Palette[2][paletteIdx] - pSrc[2][posC];
            absError += temp * temp;
            absError = (absError + round) >> localAdjC;
          }
          else
          {
            absError = 0;
          }
          temp = Palette[0][paletteIdx] - pSrc[0][pos];
          absError += (temp * temp) >> distAdjY;
          indError[paletteIdx] = discardChroma ? (temp * temp) >> distAdjY : absError;
        }

        if (absError < minError)
        {
          bestIdx = paletteIdx;
          minError = absError;
        }

        paletteIdx++;
      }
      m_indexBlock[pos] = bestIdx;

      UInt errorTemp;

      if( discardChroma && bLossless && minError != MAX_UINT )
      {
        minError = 0;
      }

      if (minError > errorLimit || calcErroBits)
      {
        Double rdCost = MAX_DOUBLE;
        if (pcCU->getCUTransquantBypass(0))
        {
          errorTemp = 0;
          UInt uiNumTotalBits = pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA);
          if (!discardChroma) uiNumTotalBits += pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_CHROMA)<<1;

          rdCost += pcCost->getLambda() * uiNumTotalBits;
          if (minError > errorLimit)
          {
            m_indexBlock[pos] -= maxSpsPaletteSize;
            useEscapeFlag = 1;
          }
        }
        else
        {
          Pel pOrg[3] = { pSrc[0][pos], 0, 0 };
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            pOrg[1] = pSrc[1][posC];
            pOrg[2] = pSrc[2][posC];
          }
          rdCost = xCalcPixelPredRD(pcCU, pOrg, pcCost, &errorTemp, discardChroma);

          if (rdCost < minError && minError > errorLimit)
          {
            m_indexBlock[pos] -= maxSpsPaletteSize;
            useEscapeFlag = 1;
          }
        }
        indError[MAX_PALETTE_SIZE - 1] = (UInt)rdCost;
        indError[MAX_PALETTE_SIZE] = (UInt)errorTemp;
      }
      m_posBlock[pos] = pos;
    }
  }

  pcCU->setPaletteEscapeSubParts(0, useEscapeFlag, 0, pcCU->getDepth(0));
  pcCU->setPaletteEscapeSubParts(1, useEscapeFlag, 0, pcCU->getDepth(0));
  pcCU->setPaletteEscapeSubParts(2, useEscapeFlag, 0, pcCU->getDepth(0));
}

Void TEncSearch::xPreCalcPaletteIndex(TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt paletteSize)
{
  Bool bLossless = pcCU->getCUTransquantBypass(0);
  Int errorLimit = bLossless ? 0 : 3 * m_paletteErrLimit;
  UInt pos;
  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

  UInt bestIdx = 0;
  UChar useEscapeFlag=0;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt uiPosC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      UInt paletteIdx = 0;
      UInt minError = MAX_UINT;
      while (paletteIdx < paletteSize)
      {
        UInt absError = MAX_UINT;
        if ( bLossless )
        {
          absError = abs( Palette[0][paletteIdx] - pSrc[0][pos] ) + abs( Palette[1][paletteIdx] - pSrc[1][uiPosC] ) + abs( Palette[2][paletteIdx] - pSrc[2][uiPosC] );
        }
        else
        {
          absError = ( abs(Palette[0][paletteIdx] - pSrc[0][pos])  >> DISTORTION_PRECISION_ADJUSTMENT(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]  -8) )
                     + ( abs(Palette[1][paletteIdx] - pSrc[1][uiPosC]) >> DISTORTION_PRECISION_ADJUSTMENT(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA]-8) )
                     + ( abs(Palette[2][paletteIdx] - pSrc[2][uiPosC]) >> DISTORTION_PRECISION_ADJUSTMENT(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA]-8) );
        }
        if (absError < minError)
        {
          bestIdx = paletteIdx;
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }
      m_indexBlock[pos] = bestIdx;
      if (minError > errorLimit)
      {
        m_indexBlock[pos] -= pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();
        useEscapeFlag=1;
      }
    }
  }

  pcCU->setPaletteEscapeSubParts(0, useEscapeFlag,0, pcCU->getDepth(0));
  pcCU->setPaletteEscapeSubParts(1, useEscapeFlag,0, pcCU->getDepth(0));
  pcCU->setPaletteEscapeSubParts(2, useEscapeFlag,0, pcCU->getDepth(0));
}

Void TEncSearch::xReorderPalette(TComDataCU* pcCU, Pel *pPalette[3], UInt numComp)
{
  UInt paletteSizePrev, uiDictMaxSize;
  Pel * pPalettePrev[3];
  UInt maxPaletteSize = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();
  UInt maxPalettePredSize = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize();
  Pel* pPaletteTemp[3];
  for (UInt ch = 0; ch < 3; ch++)
  {
    pPaletteTemp[ch] = (Pel*)xMalloc(Pel, maxPaletteSize);
  }
  ComponentID compBegin = COMPONENT_Y;

  for (UInt comp = compBegin; comp < compBegin + numComp; comp++)
  {
    pPalettePrev[comp] = pcCU->getPalettePred(pcCU, comp, paletteSizePrev);
    for (UInt i = 0; i < maxPaletteSize; i++)
    {
      pPaletteTemp[comp][i] = pPalette[comp][i];
    }
  }

  uiDictMaxSize = pcCU->getPaletteSize(compBegin, 0);

  UInt idxPrev = 0, idxCurr = 0;
  Bool bReused = false;
  Bool *bPredicted, *bReusedPrev;
  bPredicted  = (Bool*)xMalloc(Bool, maxPaletteSize + 1);
  bReusedPrev = (Bool*)xMalloc(Bool, maxPalettePredSize + 1);
  memset(bPredicted, 0, sizeof(Bool)*(maxPaletteSize + 1));
  memset(bReusedPrev, 0, sizeof(Bool)*(maxPalettePredSize + 1));

  Int numPaletteRceived = uiDictMaxSize;
  UInt numPalettePredicted = 0;

  for (idxCurr = 0; idxCurr < uiDictMaxSize; idxCurr++)
  {
    bReused = false;
    Int iCounter = 0;

    for (idxPrev = 0; idxPrev < paletteSizePrev; idxPrev++)
    {
      iCounter = 0;

      for (UInt comp = compBegin; comp < compBegin + numComp; comp++)
      {
        if (pPalettePrev[comp][idxPrev] == pPalette[comp][idxCurr])
        {
          iCounter++;
        }
      }
      if (iCounter == numComp)
      {
        bReused = true;
        break;
      }
    }
    bReusedPrev[idxPrev] = bReused;
    bPredicted[idxCurr] = bReused;
    if (bPredicted[idxCurr])
    {
      numPaletteRceived--;
      numPalettePredicted++;
    }
  }

  assert( numPaletteRceived >= 0 );
  assert( numPalettePredicted <= uiDictMaxSize );

  for (idxPrev = 0; idxPrev < maxPalettePredSize; idxPrev++)
  {
    for (UInt comp = compBegin; comp < compBegin + numComp; comp++)
    {
      pcCU->setPrevPaletteReusedFlagSubParts(comp, bReusedPrev[idxPrev], idxPrev, 0, pcCU->getDepth(0));
    }
  }
  idxCurr = 0;
  for (UInt prevIdx = 0; prevIdx < paletteSizePrev; prevIdx++)
  {
    if (bReusedPrev[prevIdx])
    {
      for (UInt comp = compBegin; comp < compBegin + numComp; comp++)
      {
        pPalette[comp][idxCurr] = pPalettePrev[comp][prevIdx];
      }
      idxCurr++;
    }
  }

  for (UInt idx = 0; idx < uiDictMaxSize; idx++)
  {
    if (bPredicted[idx] == 0)
    {
      for (UInt comp = compBegin; comp < compBegin + numComp; comp++)
      {
        pPalette[comp][idxCurr] = pPaletteTemp[comp][idx];
      }
      idxCurr++;
    }
  }
  for (UInt ch = 0; ch < 3; ch++)
  {
    if (pPaletteTemp[ch])
    {
      xFree(pPaletteTemp[ch]);
      pPaletteTemp[ch] = NULL;
    }
  }
  if (bPredicted)
  {
    xFree(bPredicted);
    bPredicted = NULL;
  }
  if (bReusedPrev)
  {
    xFree(bReusedPrev);
    bReusedPrev = NULL;
  }
}

UInt TEncSearch::xFindCandidatePalettePredictors(UInt paletteIndBest[], TComDataCU* pcCU, Pel *Palette[3], Pel* pPred[3], UInt paletteSizeTemp, UInt maxNoPredInd)
{
  UInt absError=0, minError;
  UInt palettePredError[MAX_PALETTE_PRED_SIZE];
#if !FULL_NBIT
  BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
#endif

  for(Int t = 0; t < pcCU->getLastPaletteInLcuSizeFinal(0); t++)
  {
    absError=0;
    Int iTemp=pPred[0][t] - Palette[0][paletteSizeTemp];
    absError += (iTemp * iTemp) >> DISTORTION_PRECISION_ADJUSTMENT((bitDepths.recon[CHANNEL_TYPE_LUMA] - 8) << 1);
    if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
    {
      iTemp = pPred[1][t] - Palette[1][paletteSizeTemp];
      absError += (iTemp * iTemp) >> DISTORTION_PRECISION_ADJUSTMENT((bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8) << 1);
      iTemp = pPred[2][t] - Palette[2][paletteSizeTemp];
      absError += (iTemp * iTemp) >> DISTORTION_PRECISION_ADJUSTMENT((bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8) << 1);
    }

    palettePredError[t] = absError;
    paletteIndBest[t] = t;
  }

  UInt bestInd;
  for(Int t=0; t < maxNoPredInd; t++)
  {
    bestInd = t;
    minError = palettePredError[t];

    for (UInt l=t+1; l < pcCU->getLastPaletteInLcuSizeFinal(0); l++)
    {
      if (palettePredError[l] < minError)
      {
        bestInd=l;
        minError=palettePredError[l];
      }
    }

    swap(palettePredError[bestInd], palettePredError[t]);
    swap(paletteIndBest[bestInd], paletteIndBest[t]);
  }

  UInt maxPredCheck=min((UInt)pcCU->getLastPaletteInLcuSizeFinal(0), maxNoPredInd);

  return(maxPredCheck);
}

Void TEncSearch::xDerivePaletteLossy( TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost )
{
  Int errorLimit = m_paletteErrLimit;
  UInt totalSize = height*width;
  SortingElement *psList = new SortingElement [totalSize];
  SortingElement sElement;
  UInt dictMaxSize = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();
  SortingElement *pListSort = new SortingElement [dictMaxSize + 1];
  UInt idx = 0;
  UInt pos;
  Int last = -1;
  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

  SortingElement *psListHistogram = new SortingElement[totalSize];
  SortingElement *psInitial = new SortingElement[totalSize];
  UInt hisIdx = 0;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
      {
        sElement.setAll(pSrc[0][pos], 0, 0);
      }
      else
      {
        sElement.setAll(pSrc[0][pos], pSrc[1][posC], pSrc[2][posC]);
      }
      Int i = 0;
      for (i = hisIdx - 1; i >= 0; i--)
      {
        if (psListHistogram[i].EqualData(sElement))
        {
          psListHistogram[i].addElement(sElement);
          break;
        }
      }
      if (i == -1)
      {
        psListHistogram[hisIdx].copyDataFrom(sElement);
        psListHistogram[hisIdx].cnt = 1;
        hisIdx++;
      }
    }
  }

  UInt hisCnt, maxIdx;
  UInt limit = ((height << 2)*errorLimit) >> 7;
  limit = (limit > (height >> 1)) ? limit : (height >> 1);

  Bool bOtherPeakExist;
  while (true)
  {
    hisCnt = psListHistogram[0].cnt;
    maxIdx = 0;
    for (UInt j = 1; j < hisIdx; j++)
    {
      if (psListHistogram[j].cnt >= hisCnt)
      {
        hisCnt = psListHistogram[j].cnt;
        maxIdx = j;
      }
    }

    if (hisCnt >= limit)
    {
      bOtherPeakExist = false;
      for (UInt j = 0; j < hisIdx; j++)
      {
        if (psListHistogram[j].cnt >= (hisCnt >> 1) && j != maxIdx)
        {
          if (psListHistogram[maxIdx].almostEqualData(psListHistogram[j], errorLimit >> 2, pcCU->getSlice()->getSPS()->getBitDepths()))
          {
            bOtherPeakExist = true;
          }
        }
      }

      if (!bOtherPeakExist)
      {
        psList[idx].copyAllFrom(psListHistogram[maxIdx]);
        psInitial[idx].copyAllFrom(psListHistogram[maxIdx]);
        last = idx;
        idx++;

        for (UInt j = 0; j < hisIdx; j++)
        {
          if (psListHistogram[maxIdx].almostEqualData(psListHistogram[j], errorLimit >> 2, pcCU->getSlice()->getSPS()->getBitDepths()) && j != maxIdx)
          {
            psListHistogram[j].ResetElement();
          }
        }
      }

      psListHistogram[maxIdx].ResetElement();
    }
    else
    {
      break;
    }
  }

  UInt initialIdx = idx;
  Bool bMatched;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
      {
        sElement.setAll(pSrc[0][pos], 0, 0);
      }
      else
      {
        sElement.setAll(pSrc[0][pos], pSrc[1][posC], pSrc[2][posC]);
      }
      bMatched = false;
      for (Int i = 0; i < initialIdx; i++)
      {
        bMatched |= psInitial[i].EqualData(sElement);
      }

      if (!bMatched)
      {
        Int besti = last, bestSAD = (last == -1) ? MAX_UINT : psList[last].getSAD(sElement, pcCU->getSlice()->getSPS()->getBitDepths());
        if (bestSAD)
        {
          for (Int i = idx - 1; i >= 0; i--)
          {
            UInt sad = psList[i].getSAD(sElement, pcCU->getSlice()->getSPS()->getBitDepths());
            if (sad < bestSAD)
            {
              bestSAD = sad;
              besti = i;
              if (!sad) break;
            }
          }
        }

        if (besti >= 0 && psList[besti].almostEqualData(sElement, errorLimit, pcCU->getSlice()->getSPS()->getBitDepths()))
        {
          psList[besti].addElement(sElement);
          last = besti;
        }
        else
        {
          psList[idx].copyDataFrom(sElement);
          psList[idx].cnt = 1;
          last = idx;
          idx++;
        }
      }
    }
  }

  for (Int i = 0; i < dictMaxSize; i++)
  {
    pListSort[i].cnt  = 0;
    pListSort[i].setAll(0, 0, 0) ;
  }

  //bubble sorting
  dictMaxSize = 1;
  for (Int i = 0; i < idx; i++)
  {
    if( psList[i].cnt > pListSort[dictMaxSize-1].cnt )
    {
      Int j;
      for (j = dictMaxSize; j > 0; j--)
      {
        if (psList[i].cnt > pListSort[j-1].cnt)
        {
          pListSort[j].copyAllFrom (pListSort[j-1]);
          dictMaxSize = std::min(dictMaxSize + 1, pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize());
        }
        else
        {
          break;
        }
      }
      pListSort[j].copyAllFrom (psList[i]);
    }
  }

  Int paletteIndPred[MAX_PALETTE_SIZE];
  memset(paletteIndPred, 0, MAX_PALETTE_SIZE*sizeof(Int));

  paletteSize = 0;
  Pel *pPred[3]  = { pcCU->getLastPaletteInLcuFinal(0), pcCU->getLastPaletteInLcuFinal(1), pcCU->getLastPaletteInLcuFinal(2) };
  UInt uiNumTotalBits = pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) + (pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_CHROMA)<<1);
  Double bitCost = pcCost->getLambda() * uiNumTotalBits;
#if !FULL_NBIT
  BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
#endif
  for (Int i = 0; i < pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize(); i++)
  {
    if( pListSort[i].cnt )
    {
      Int half = pListSort[i].cnt>>1;
      Palette[0][paletteSize] = (pListSort[i].sumData[0]+half)/pListSort[i].cnt;
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        Palette[1][paletteSize] = (pListSort[i].sumData[1] + half) / pListSort[i].cnt;
        Palette[2][paletteSize] = (pListSort[i].sumData[2] + half) / pListSort[i].cnt;
      }

      Int best = -1;
      if( errorLimit )
      {
        Double pal[3] = { pListSort[i].sumData[0]/(Double)pListSort[i].cnt,
                          pListSort[i].sumData[1]/(Double)pListSort[i].cnt,
                          pListSort[i].sumData[2]/(Double)pListSort[i].cnt };

        Double err      = pal[0] - Palette[0][paletteSize];
        Double bestCost = (err*err) / ( 1<<(2*DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA]-8)) );
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          err = pal[1] - Palette[1][paletteSize]; bestCost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
          err = pal[2] - Palette[2][paletteSize]; bestCost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
        }
        bestCost = bestCost * pListSort[i].cnt + bitCost;

        for(Int t=0; t<pcCU->getLastPaletteInLcuSizeFinal(0); t++)
        {
          err = pal[0] - pPred[0][t];
          Double cost = (err*err) / ( 1<<(2*DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA]-8)) );
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            err = pal[1] - pPred[1][t]; cost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
            err = pal[2] - pPred[2][t]; cost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
          }
          cost *= pListSort[i].cnt;
          if(cost < bestCost)
          {
            best = t;
            bestCost = cost;
          }
        }
        if( best != -1 )
        {
          Palette[0][paletteSize] = pPred[0][best];
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            Palette[1][paletteSize] = pPred[1][best];
            Palette[2][paletteSize] = pPred[2][best];
          }
        }
        paletteIndPred[paletteSize]=best;
      }

      Bool bDuplicate = false;
      if( pListSort[i].cnt == 1 && best == -1 )
      {
        bDuplicate = true;
      }
      else
      {
        for( Int t=0; t<paletteSize; t++)
        {
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
          {
            if (Palette[0][paletteSize] == Palette[0][t])
            {
              bDuplicate = true;
              break;
            }
          }
          else
          {
            if (Palette[0][paletteSize] == Palette[0][t] && Palette[1][paletteSize] == Palette[1][t] && Palette[2][paletteSize] == Palette[2][t])
            {
              bDuplicate = true;
              break;
            }
          }
        }
      }
      if (!bDuplicate)
      {
        paletteSize++;
      }
    }
    else
    {
      break;
    }
  }

  UInt palettePredSamples[MAX_PALETTE_SIZE][5];
  memset(palettePredSamples, 0, sizeof(palettePredSamples));
  Int errorLimitSqr = 3 * m_paletteErrLimit * m_paletteErrLimit;

  UInt absError;
  UInt minError;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      UInt bestIdx=0, paletteIdx = 0;
      Bool discardChroma = y&scaleY || x&scaleX || pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400;
      minError = MAX_UINT;

      while (paletteIdx < paletteSize)
      {
        Int temp=Palette[0][paletteIdx] - pSrc[0][pos];
        absError = (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          temp = Palette[1][paletteIdx] - pSrc[1][posC];
          absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
          temp = Palette[2][paletteIdx] - pSrc[2][posC];
          absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
        }

        if (absError < minError)
        {
          bestIdx = paletteIdx;
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }

      UInt escape=0;
      if (minError > errorLimitSqr)
      {
        Pel pOrg[3]={ pSrc[0][pos], 0, 0};
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          pOrg[1] = pSrc[1][posC];
          pOrg[2] = pSrc[2][posC];
        }
        UInt errorTemp;
        Double rdCost = xCalcPixelPredRD(pcCU, pOrg, pcCost, &errorTemp);
        if (rdCost<minError)
        {
          escape=1;
        }
      }

      if (escape==0)
      {
        palettePredSamples[bestIdx][0]++;
        palettePredSamples[bestIdx][1] += pSrc[0][pos];
        if (!discardChroma)
        {
          palettePredSamples[bestIdx][2] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[1][posC];
          palettePredSamples[bestIdx][3] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[2][posC];
          palettePredSamples[bestIdx][4] += SCM_V0034_PALETTE_CHROMA_SETTINGS;
        }
        else if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          palettePredSamples[bestIdx][2] += pSrc[1][posC];
          palettePredSamples[bestIdx][3] += pSrc[2][posC];
          palettePredSamples[bestIdx][4]++;
        }
        m_indexBlock[pos] = bestIdx;
      }
      else
      {
        m_indexBlock[pos]=-1;
      }
    }
  }

  UInt paletteIndBest[MAX_PALETTE_PRED_SIZE];

  UInt   paletteSizeTemp=0;
  for (Int i = 0; i < paletteSize; i++)
  {
    if(palettePredSamples[i][0] > 0)
    {
      Palette[0][paletteSizeTemp] = (palettePredSamples[i][1]+palettePredSamples[i][0]/2)/palettePredSamples[i][0];
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        Palette[1][paletteSizeTemp] = (palettePredSamples[i][2] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
        Palette[2][paletteSizeTemp] = (palettePredSamples[i][3] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
      }

      Double dMinError = pcCost->getLambda()*(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]+2*pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA]);

      for (UInt y = 0; y < height; y++)
      {
        for (UInt x = 0; x < width; x++)
        {
          pos = y * width + x;
          if (m_indexBlock[pos]==i)
          {
            UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

            Int temp=Palette[0][paletteSizeTemp] - pSrc[0][pos];
            dMinError += (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
            if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
            {
              temp = Palette[1][paletteSizeTemp] - pSrc[1][posC];
              dMinError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
              temp = Palette[2][paletteSizeTemp] - pSrc[2][posC];
              dMinError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
            }
          }
        }
      }

      UInt maxPredCheck=xFindCandidatePalettePredictors(paletteIndBest, pcCU, Palette, pPred, paletteSizeTemp, MAX_PRED_CHEK);

      Int best=-1;
      if (paletteIndPred[i]>=0)
      {
        for (int t=0; t<maxPredCheck; t++)
        {
          if (paletteIndPred[i]==paletteIndBest[t])
          {
            best=1;
          }
        }
        if (best==-1)
        {
          paletteIndBest[maxPredCheck]=paletteIndPred[i];
          maxPredCheck++;
        }
      }

      best=-1;
      UInt testedPalettePred;

      for(int t=0; t<maxPredCheck; t++)
      {
        testedPalettePred=paletteIndBest[t];

        absError=0;
        for (UInt y = 0; y < height; y++)
        {
          for (UInt x = 0; x < width; x++)
          {
            pos = y * width + x;
            if (m_indexBlock[pos]==i)
            {
              UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
              Int temp=pPred[0][testedPalettePred] - pSrc[0][pos];
              absError += (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
              if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
              {
                temp = pPred[1][testedPalettePred] - pSrc[1][posC];
                absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
                temp = pPred[2][testedPalettePred] - pSrc[2][posC];
                absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
              }
            }
          }
          if (absError>dMinError)
          {
            break;
          }
        }

        if (absError < dMinError || (absError == dMinError && best>testedPalettePred))
        {
          best = testedPalettePred;
          dMinError = absError;
        }
      }

      if( best != -1 )
      {
        Palette[0][paletteSizeTemp] = pPred[0][best];
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          Palette[1][paletteSizeTemp] = pPred[1][best];
          Palette[2][paletteSizeTemp] = pPred[2][best];
        }
      }

      Bool bDuplicate = false;
      if( palettePredSamples[i][0] == 1 && best == -1 )
      {
        bDuplicate = true;
      }
      else
      {
        for( Int t=0; t<paletteSizeTemp; t++)
        {
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
          {
            if (Palette[0][paletteSizeTemp] == Palette[0][t])
            {
              bDuplicate = true;
              break;
            }
          }
          else if( Palette[0][paletteSizeTemp] == Palette[0][t] && Palette[1][paletteSizeTemp] == Palette[1][t] && Palette[2][paletteSizeTemp] == Palette[2][t] )
          {
            bDuplicate = true;
            break;
          }
        }
      }
      if (!bDuplicate)
      {
        paletteSizeTemp++;
      }
    }
  }

  paletteSize=paletteSizeTemp;

  delete[] psList;
  delete[] pListSort;

  delete[] psListHistogram;
  delete[] psInitial;
}

Void TEncSearch::xDerivePaletteLossyIterative(TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3],  UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost)
{
  UInt pos, paletteIdx = 0, bestIdx;

  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

  Int distAdjY = DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA] - 8) << 1);
  Int distAdjC = DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1);
  UInt palettePredSamples[MAX_PALETTE_SIZE][5], noSamples[MAX_PALETTE_SIZE];
  memset(palettePredSamples, 0, sizeof(palettePredSamples));
  Int errorLimitSqr = pcCU->getCUTransquantBypass(0) ? 0 : 3 * m_paletteErrLimit * m_paletteErrLimit;

  Pel *pPred[3]  = { pcCU->getLastPaletteInLcuFinal(0), pcCU->getLastPaletteInLcuFinal(1), pcCU->getLastPaletteInLcuFinal(2) };
  Pel pPaletteTemp[3][MAX_PALETTE_SIZE];

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;

      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      Int  localAdjC = distAdjC;
      Bool discardChroma = y&scaleY || x&scaleX || pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400;
      if (discardChroma) localAdjC+=SCM_V0034_PALETTE_CHROMA_SHIFT_ADJ;

      bestIdx=0;
      UInt minError = MAX_UINT;

      paletteIdx=0;
      while (paletteIdx < paletteSize)
      {
        Int temp=Palette[0][paletteIdx] - pSrc[0][pos];
        UInt absError = ( temp * temp ) >> distAdjY;
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          temp = Palette[1][paletteIdx] - pSrc[1][posC];
          absError += (temp * temp) >> localAdjC;
          temp = Palette[2][paletteIdx] - pSrc[2][posC];
          absError += (temp * temp) >> localAdjC;
        }

        if (absError < minError)
        {
          bestIdx = paletteIdx;
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }

      UInt escape=0;
      if (minError > errorLimitSqr)
      {
        UInt errorTemp;
        Pel pOrg[3]={ pSrc[0][pos], 0, 0};
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          pOrg[1] = pSrc[1][posC];
          pOrg[2] = pSrc[2][posC];
        }
        Double rdCost = xCalcPixelPredRD(pcCU, pOrg, pcCost, &errorTemp, discardChroma);

        if (rdCost<minError)
        {
          escape=1;
        }
      }

      if (escape==0)
      {
        palettePredSamples[bestIdx][0]++;
        palettePredSamples[bestIdx][1] += pSrc[0][pos];
        if (!discardChroma)
        {
          palettePredSamples[bestIdx][2] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[1][posC];
          palettePredSamples[bestIdx][3] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[2][posC];
          palettePredSamples[bestIdx][4] += SCM_V0034_PALETTE_CHROMA_SETTINGS;
        }
        else if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          palettePredSamples[bestIdx][2] += pSrc[1][posC];
          palettePredSamples[bestIdx][3] += pSrc[2][posC];
          palettePredSamples[bestIdx][4]++;
        }
        m_indexBlock[pos] = bestIdx;
      }
      else
      {
        m_indexBlock[pos]=-1;
      }
    }
  }

  UInt paletteIndBest[MAX_PALETTE_PRED_SIZE];
  UInt   paletteSizeTemp=0;
  for (Int i = 0; i < paletteSize; i++)
  {
    if(palettePredSamples[i][0]>0)
    {
      pPaletteTemp[0][paletteSizeTemp] = (palettePredSamples[i][1]+palettePredSamples[i][0]/2)/palettePredSamples[i][0];
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        pPaletteTemp[1][paletteSizeTemp] = (palettePredSamples[i][2] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
        pPaletteTemp[2][paletteSizeTemp] = (palettePredSamples[i][3] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
      }

      noSamples[paletteSizeTemp]=palettePredSamples[i][0];

      Double minError = pcCost->getLambda()*(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA] + 2 * pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA]);
      Double absError;

      for (UInt y = 0; y < height; y++)
      {
        for (UInt x = 0; x < width; x++)
        {
          pos = y * width + x;
          if (m_indexBlock[pos]==i)
          {
            UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
            Int temp=pPaletteTemp[0][paletteSizeTemp] - pSrc[0][pos];
            Int  localAdjC = distAdjC;
            if (y&scaleY || x&scaleX) localAdjC+=SCM_V0034_PALETTE_CHROMA_SHIFT_ADJ;
            minError += ( temp * temp ) >> distAdjY;
            if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
            {
              temp = pPaletteTemp[1][paletteSizeTemp] - pSrc[1][posC];
              minError += (temp * temp) >> localAdjC;
              temp = pPaletteTemp[2][paletteSizeTemp] - pSrc[2][posC];
              minError += (temp * temp) >> localAdjC;
            }
          }
        }
      }

      UInt maxPredCheck=xFindCandidatePalettePredictors(paletteIndBest, pcCU, Palette, pPred, paletteSizeTemp, MAX_PRED_CHEK);
      Int best=-1;
      UInt testedPalettePred;

      for(int t=0; t<maxPredCheck; t++)
      {
        testedPalettePred=paletteIndBest[t];

        absError=0;
        for (UInt y = 0; y < height; y++)
        {
          for (UInt x = 0; x < width; x++)
          {
            pos = y * width + x;
            if (m_indexBlock[pos]==i)
            {
              UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

              Int temp=pPred[0][testedPalettePred] - pSrc[0][pos];
              Int  localAdjC = distAdjC;
              if (y&scaleY || x&scaleX) localAdjC+=SCM_V0034_PALETTE_CHROMA_SHIFT_ADJ;
              absError += ( temp * temp ) >> distAdjY;
              if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
              {
                temp = pPred[1][testedPalettePred] - pSrc[1][posC];
                absError += (temp * temp) >> localAdjC;
                temp = pPred[2][testedPalettePred] - pSrc[2][posC];
                absError += (temp * temp) >> localAdjC;
              }
            }
          }
          if (absError>minError)
          {
            break;
          }
        }

        if (absError < minError || (absError == minError && best>testedPalettePred))
        {
          best = testedPalettePred;
          minError = absError;
        }
      }

      if( best != -1 )
      {
        pPaletteTemp[0][paletteSizeTemp] = pPred[0][best];
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          pPaletteTemp[1][paletteSizeTemp] = pPred[1][best];
          pPaletteTemp[2][paletteSizeTemp] = pPred[2][best];
        }
      }

      Bool bDuplicate = false;
      if( palettePredSamples[i][0] == 1 && best == -1 )
      {
        bDuplicate = true;
      }
      else
      {
        for( Int t=0; t<paletteSizeTemp; t++)
        {
          if( pPaletteTemp[0][paletteSizeTemp] == pPaletteTemp[0][t] && pPaletteTemp[1][paletteSizeTemp] == pPaletteTemp[1][t] && pPaletteTemp[2][paletteSizeTemp] == pPaletteTemp[2][t] )
          {
            bDuplicate = true;
            break;
          }
        }
      }
      if( !bDuplicate ) paletteSizeTemp++;
    }
  }

  paletteSize=paletteSizeTemp;

  for (paletteIdx=0; paletteIdx<paletteSize; paletteIdx++)
  {
    bestIdx=paletteIdx;
    UInt maxSamples=noSamples[paletteIdx];

    for (UInt paletteIdxSec=paletteIdx+1; paletteIdxSec<paletteSize; paletteIdxSec++)
    {
      if (noSamples[paletteIdxSec]>maxSamples)
      {
        bestIdx=paletteIdxSec;
        maxSamples=noSamples[paletteIdxSec];
      }
    }

    Palette[0][paletteIdx]=pPaletteTemp[0][bestIdx];
    if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
    {
      Palette[1][paletteIdx] = pPaletteTemp[1][bestIdx];
      Palette[2][paletteIdx] = pPaletteTemp[2][bestIdx];
    }

    pPaletteTemp[0][bestIdx]=pPaletteTemp[0][paletteIdx];
    if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
    {
      pPaletteTemp[1][bestIdx] = pPaletteTemp[1][paletteIdx];
      pPaletteTemp[2][bestIdx] = pPaletteTemp[2][paletteIdx];
    }

    noSamples[bestIdx]=noSamples[paletteIdx];
  }
}

Void TEncSearch::xDerivePaletteLossless(TComDataCU* pcCU, Pel *Palette[3], Pel* pSrc[3], UInt width, UInt height, UInt &paletteSize)
{
  std::vector<SortingElement> psList;
  SortingElement sElement;
  Int idx = 0;
  UInt pos;

  const UInt maxPaletteSizeSPS = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();
  paletteSize = 0;

  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

  idx = 0;
  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      Bool discardChroma = y&scaleY || x&scaleX || pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400;
      Int  defIdx = -1, defSAD = MAX_INT;
      Int  discIdx = -1, discSAD = MAX_INT;

      Int i = 0;
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
      {
        sElement.setAll(pSrc[0][pos], 0, 0);
      }
      else
      {
        sElement.setAll(pSrc[0][pos], pSrc[1][posC], pSrc[2][posC]);
      }
      for (i = idx - 1; i >= 0; i--)
      {
        if( psList[i].data[0] == sElement.data[0] && psList[i].data[1] == sElement.data[1] && psList[i].data[2] == sElement.data[2] )
        {
          psList[i].cnt++;
          if ( !discardChroma )
          {
            psList[i].lastCnt = 1;
          }
          break;
        }

        if( (scaleX||scaleY) && psList[i].data[0] == sElement.data[0] && pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400 )
        {
          Int sad = abs(psList[i].data[1] - pSrc[1][posC]) + abs(psList[i].data[2] - pSrc[2][posC]);
          if( !discardChroma && !psList[i].lastCnt && sad < discSAD )
          {
            discIdx = i;
            discSAD = sad;
          }
          if( discardChroma && sad < defSAD )
          {
            defIdx = i;
            defSAD = sad;
          }
        }
      }
      if( discIdx >= 0 )
      {
        psList[discIdx].cnt++;
        psList[discIdx].setAll(pSrc[0][pos], pSrc[1][posC], pSrc[2][posC]);
        psList[discIdx].lastCnt = 1;
      }
      else if( i == -1 && defIdx >= 0 )
      {
        psList[defIdx].cnt++;
      }
      else if (i == -1)
      {
        psList.push_back(sElement);
        psList[idx].cnt++;
        psList[idx].lastCnt = discardChroma ? 0 : 1;
        idx++;
      }
    }
  }

  //insertion sort, high frequency -> low frequency
  std::stable_sort(psList.begin(), psList.end());
  UInt paletteSizePrev;
  Pel *pPalettePrev[3];
  for (UInt comp = 0; comp < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++)
  {
    pPalettePrev[comp] = pcCU->getPalettePred(pcCU, comp, paletteSizePrev);
  }

  if( paletteSize < maxPaletteSizeSPS )
  {
    for (Int i = 0; i < idx; i++)
    {
      Bool includeIntoPalette = true;
      if( (scaleX||scaleY) && psList[i].cnt > 0 && !psList[i].lastCnt && pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400) // Find if it can be replaced
      {
        Int bestCand = -1;
        for( UInt idxPrev = 0; idxPrev < paletteSizePrev; idxPrev++ )
        {
          if( psList[i].data[0] == pPalettePrev[0][idxPrev] )
          {
            bestCand = idxPrev;
            break;
          }
        }

        if( bestCand != -1 )
        {
          psList[i].data[1] = pPalettePrev[1][bestCand];
          psList[i].data[2] = pPalettePrev[2][bestCand];
        }
        else if (psList[i].cnt < 3)
        {
          includeIntoPalette = false;
        }
      }
      else if( psList[i].cnt == 1 )
      {
        includeIntoPalette = false;
        for( UInt idxPrev = 0; idxPrev < paletteSizePrev; idxPrev++ )
        {
          UInt iCounter = 0;

          for( UInt comp = 0; comp < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++ )
          {
            if( psList[i].data[comp] == pPalettePrev[comp][idxPrev] )
            {
              iCounter++;
            }
            else
            {
              break;
            }
          }
          if( iCounter == 3 )
          {
            includeIntoPalette = true;
            break;
          }
          else if( pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 && iCounter == 1 )
          {
            includeIntoPalette = true;
            break;
          }
        }
      }

      if( includeIntoPalette && psList[i].cnt)
      {
        Palette[0][paletteSize] = psList[i].data[0];
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          Palette[1][paletteSize] = psList[i].data[1];
          Palette[2][paletteSize] = psList[i].data[2];
        }
        paletteSize++;
        if (paletteSize == pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize())
        {
          break;
        }
      }
    }
  }
}

Void TEncSearch::xCalcPixelPred(TComDataCU* pcCU, Pel* pOrg [3], Pel*paPixelValue[3], Pel * paRecoValue[3], UInt width, UInt strideOrg, UInt startPos)
{
  Bool bLossless = pcCU->getCUTransquantBypass (0);
  Int iQP[3];
  Int iQPrem[3];
  Int iQPper[3];
  Int quantiserScale[3];
  Int quantiserRightShift[3];
  Int rightShiftOffset[3];
  Int invquantiserRightShift[3];
  Int iAdd[3];
  TCoeff iTmpValue = 0;
  for (UInt ch = 0; ch < MAX_NUM_COMPONENT; ch++)
  {
    assert(!pcCU->getColourTransform(0));
    QpParam cQP(*pcCU, ComponentID(ch), 0);
    iQP[ch] = cQP.Qp;
    iQPrem[ch] = iQP[ch] % 6;
    iQPper[ch] = iQP[ch] / 6;
    quantiserScale[ch] = g_quantScales[iQPrem[ch]];
    quantiserRightShift[ch] = QUANT_SHIFT + iQPper[ch];
    rightShiftOffset[ch] = 1 << (quantiserRightShift[ch] - 1);
    invquantiserRightShift[ch] = IQUANT_SHIFT;
    iAdd[ch] = 1 << (invquantiserRightShift[ch] - 1);
  }

  UInt y, x;
  y = startPos / width;
  x = startPos % width;
  UInt scanIdx = y * strideOrg + x;
  UInt YIdxRaster = pcCU->getPaletteScanRotationModeFlag(0)? (x * strideOrg + y) : (y * strideOrg + x);
  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);
  UInt xC, yC, scanIdxC, YIdxRasterC;
  if(!pcCU->getPaletteScanRotationModeFlag(0))
  {
    xC = (x>>scaleX);
    yC = (y>>scaleY);
    scanIdxC = yC * (strideOrg>>scaleX) + xC;
    YIdxRasterC = yC * (strideOrg>>scaleX) + xC;
  }
  else
  {
    xC = (x>>scaleY);
    yC = (y>>scaleX);
    scanIdxC = yC * (strideOrg>>scaleY) + xC;
    YIdxRasterC = xC * (strideOrg>>scaleX) + yC;
  }

  if (bLossless)
  {
    for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch ++)
    {
      if( ch == 0 )
      {
        paPixelValue[ch][scanIdx] =  pOrg[ch][YIdxRaster];
        paRecoValue[ch][YIdxRaster] = pOrg[ch][YIdxRaster];
      }
      else
      {
        if(   pcCU->getPic()->getChromaFormat() == CHROMA_444 ||
            ( pcCU->getPic()->getChromaFormat() == CHROMA_420 && ((x&1) == 0) && ((y&1) == 0) ) ||
            ( pcCU->getPic()->getChromaFormat() == CHROMA_422 && ((!pcCU->getPaletteScanRotationModeFlag(0) && ((x&1) == 0)) || (pcCU->getPaletteScanRotationModeFlag(0) && ((y&1) == 0))) )
          )
        {
          paPixelValue[ch][scanIdxC] =  pOrg[ch][YIdxRasterC];
          paRecoValue[ch][YIdxRasterC] = pOrg[ch][YIdxRasterC];
        }
      }
    }
  }
  else
  {
    BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
    for (UInt ch = 0; ch < (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); ch ++)
    {
      if( ch == 0 )
      {
        paPixelValue[ch][scanIdx] = Pel(max<Int>( 0, ((pOrg[ch][YIdxRaster] * quantiserScale[ch] + rightShiftOffset[ch]) >> quantiserRightShift[ch]) ));

        assert( paPixelValue[ch][scanIdx] < ( 1 << ( pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_LUMA ) + 1 ) ) );
        iTmpValue = (((paPixelValue[ch][scanIdx]*g_invQuantScales[iQPrem[ch]])<<iQPper[ch]) + iAdd[ch])>>invquantiserRightShift[ch];
        paRecoValue[ch][YIdxRaster] = Pel(ClipBD<Int>(iTmpValue, bitDepths.recon[ch? 1:0]));
      }
      else
      {
        if(   pcCU->getPic()->getChromaFormat() == CHROMA_444 ||
            ( pcCU->getPic()->getChromaFormat() == CHROMA_420 && ((x&1) == 0) && ((y&1) == 0)) ||
            ( pcCU->getPic()->getChromaFormat() == CHROMA_422 && ((!pcCU->getPaletteScanRotationModeFlag(0) && ((x&1) == 0)) || (pcCU->getPaletteScanRotationModeFlag(0) && ((y&1) == 0))) )
          )
        {
          paPixelValue[ch][scanIdxC] = Pel(max<Int>( 0, ((pOrg[ch][YIdxRasterC] * quantiserScale[ch] + rightShiftOffset[ch]) >> quantiserRightShift[ch]) ));

          assert( paPixelValue[ch][scanIdxC] < ( 1 << ( pcCU->getSlice()->getSPS()->getBitDepth( CHANNEL_TYPE_CHROMA ) + 1 ) ) );
          iTmpValue = (((paPixelValue[ch][scanIdxC]*g_invQuantScales[iQPrem[ch]])<<iQPper[ch]) + iAdd[ch])>>invquantiserRightShift[ch];
          paRecoValue[ch][YIdxRasterC] = Pel(ClipBD<Int>(iTmpValue, bitDepths.recon[ch? 1:0]));
        }
      }
    }
  }
}

UInt TEncSearch::xGetTruncatedBinBits(UInt symbol, UInt maxSymbol)
{
  UInt idxCodeBit = 0;
  UInt thresh;
  if (maxSymbol > 256)
  {
    UInt threshVal = 1 << 8;
    thresh = 8;
    while (threshVal <= maxSymbol)
    {
      thresh++;
      threshVal <<= 1;
    }
    thresh--;
  }
  else
  {
    thresh = g_uhPaletteTBC[maxSymbol];
  }

  UInt val = 1 << thresh;
  assert(val <= maxSymbol);
  assert((val << 1) > maxSymbol);
  assert(symbol < maxSymbol);
  UInt b = maxSymbol - val;
  assert(b < val);
  if (symbol < val - b)
  {
    idxCodeBit = thresh;
  }
  else
  {
    idxCodeBit = thresh+1;
  }
  return idxCodeBit;
}

UInt TEncSearch::xGetEpExGolombNumBins(UInt symbol, UInt count)
{
  //UInt bins = 0;
  UInt numBins = 0;

  while( symbol >= (UInt)(1<<count) )
  {
    //bins = 2 * bins + 1;
    numBins++;
    symbol -= 1 << count;
    count++;
  }
  //bins = 2 * bins + 0;
  numBins++;

  //bins = (bins << count) | symbol;
  numBins += count;

  assert( numBins <= 32 );

  return numBins;
}

Double TEncSearch::xCalcPixelPredRD(TComDataCU* pcCU, Pel pOrg[3], TComRdCost *pcCost, UInt *error, Bool discardChroma)
{
  Pel paPixelValue[3], paRecoValue[3];
  Int iQPcurr=Int(pcCU->getQP(0));

  Double rdCost = 0;
  UInt rdError = 0;
  if (pcCU->getCUTransquantBypass(0))
  {
    for (UInt ch = 0; ch < MAX_NUM_COMPONENT; ch++)
    {
      Int bitDepth = pcCU->getSlice()->getSPS()->getBitDepth(ch > 0 ? CHANNEL_TYPE_CHROMA : CHANNEL_TYPE_LUMA);
      rdCost += bitDepth;
    }
  }
  else
  {
    if (iQPcurr != m_prevQP)
    {
      m_prevQP = iQPcurr;
      for (UInt ch = 0; ch < MAX_NUM_COMPONENT; ch++)
      {
        Int iQP[3];
        Int iQPrem[3];
        Int iQPper[3];
        QpParam cQP(*pcCU, ComponentID(ch),0);
        iQP[ch] = cQP.Qp;
        iQPrem[ch] = iQP[ch] % 6;
        iQPper[ch] = iQP[ch] / 6;
        m_quantiserScale[ch] = g_quantScales[iQPrem[ch]];
        m_quantiserRightShift[ch] = QUANT_SHIFT + iQPper[ch];
        m_rightShiftOffset[ch] = 1 << (m_quantiserRightShift[ch] - 1);

        m_invQuantScales[ch]=g_invQuantScales[iQPrem[ch]];
        m_qpPer[ch]=iQPper[ch];

        m_maxVal[ch] = pcCU->xCalcMaxVals(pcCU, ComponentID(ch));
      }
    }

    BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
    for (UInt ch = 0; ch < (discardChroma ? 1 : MAX_NUM_COMPONENT); ch ++)
    {
      paPixelValue[ch] = Pel(Clip3<Int>( 0, m_maxVal[ch], ((pOrg[ch] * m_quantiserScale[ch] + m_rightShiftOffset[ch]) >> m_quantiserRightShift[ch]) ));
      paRecoValue[ch]= (((paPixelValue[ch]*m_invQuantScales[ch])<<m_qpPer[ch]) + 32)>>IQUANT_SHIFT;

      ChannelType comp = ch ? CHANNEL_TYPE_CHROMA : CHANNEL_TYPE_LUMA;
      paRecoValue[ch] = Pel(ClipBD<Int>(paRecoValue[ch], bitDepths.recon[comp]));

      Int iTemp = pOrg[ch] - paRecoValue[ch];
      rdError += (iTemp * iTemp) >> (DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[comp] - 8) << 1);
      rdCost += pcCost->getLambda() * xGetEscapeNumBins(paPixelValue[ch]);
    }
  }

  *error = rdError;
  rdCost += (*error);
  return (rdCost);
}

Bool TEncSearch::xCalLeftRun(TComDataCU* pcCU, Pel* pValue, UChar* pSPoint, UInt startPos, UInt total, UInt &run, UChar* pEscapeFlag)
{
  UInt idx = startPos;
  Pel *pcIndexBlock = m_indexBlock;
  while (idx < total)
  {
    UInt uiTraIdx = m_pScanOrder[idx];  //unified position variable (raster scan)
    pValue[uiTraIdx] = pcIndexBlock[uiTraIdx] < 0 ? pcIndexBlock[uiTraIdx] + pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize() : pcIndexBlock[uiTraIdx];
    Bool bMismatch = (pcIndexBlock[uiTraIdx] < 0);

    pSPoint[uiTraIdx] = PALETTE_RUN_LEFT;
    pEscapeFlag[uiTraIdx] = (pcIndexBlock[uiTraIdx] < 0)? 1: 0;
    UInt leftTraIdx = idx ? m_pScanOrder[idx - 1] : 0;
    if( idx > startPos &&
      ( ( pcIndexBlock[leftTraIdx] >= 0 && pValue[uiTraIdx] == pValue[leftTraIdx] && !bMismatch ) || ( bMismatch && pcIndexBlock[leftTraIdx] < 0 ) )
      )
    {
      run++;
    }
    else if (idx > startPos)
    {
      break;
    }
    idx++;
  }
  return true;
}

Bool TEncSearch::xCalAboveRun(TComDataCU* pcCU, Pel* pValue, UChar* pSPoint, UInt width, UInt startPos, UInt total, UInt &run, UChar* pEscapeFlag)
{
  UInt idx = startPos;
  UInt y = 0;
  Bool valid = false;
  Pel *pcIndexBlock = m_indexBlock;
  UInt traIdx = m_pScanOrder[idx];  //unified position variable (raster scan)

  y = traIdx / width;
  if( y == 0 )
  {
    return false;
  }

  while (idx < total)
  {
    UInt stride = width;
    traIdx = m_pScanOrder[idx];  //unified position variable (raster scan)

    pValue[traIdx] = pcIndexBlock[traIdx] < 0 ? pcIndexBlock[traIdx] + pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize() : pcIndexBlock[traIdx];
    Bool bMismatch = (pcIndexBlock[traIdx] < 0);

    pSPoint[traIdx] = PALETTE_RUN_ABOVE;
    pEscapeFlag[traIdx] = (pcIndexBlock[traIdx] < 0)? 1: 0;

    if ( ( pcIndexBlock[traIdx - stride] >= 0 && pValue[traIdx] == pValue[traIdx - stride] && !bMismatch ) ||
         ( bMismatch && pcIndexBlock[traIdx - stride] < 0 )
       )
    {
      run++;
      valid = true;
    }
    else
    {
      break;
    }
    idx++;
  }
  return valid;
}

Void TEncSearch::xRotationScan( Pel* pLevel, UInt width, UInt height )
{
  Pel tmpLevel;
  UInt pos = 0;
  UInt* pScanOrder = g_scanOrder[SCAN_UNGROUPED][SCAN_VER][g_aucConvertToBit[width] + 2][g_aucConvertToBit[height] + 2];

  for (UInt j = 1; j < height; j++)
  {
    pos += j;
    for (UInt i = j; i < width; i++)
    {
      tmpLevel = pLevel[pos];
      pLevel[pos] = pLevel[pScanOrder[pos]];
      pLevel[pScanOrder[pos]] = tmpLevel;
      pos++;
    }
  }
}

Void TEncSearch::xDerivePaletteLossyForcePrediction(TComDataCU *pcCU, Pel *Palette[3], Pel *pSrc[3], UInt width, UInt height, UInt &paletteSize, TComRdCost *pcCost)
{
  const Int errorLimit = m_paletteErrLimit;
  const UInt maxPaletteSizeSPS = pcCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxSize();
  const UInt totalSize = height * width;
  SortingElement *psList = new SortingElement[totalSize];
  SortingElement sElement;
  SortingElement *pListSort = new SortingElement[maxPaletteSizeSPS + 1];

  paletteSize = 0;
  UInt idx = 0, pos, bestIdx = 0;
  Int last = -1;

  UInt palettePredIndexUsed[MAX_PALETTE_PRED_SIZE];
  memset( palettePredIndexUsed, 0, sizeof(palettePredIndexUsed) );

  UChar paletteIndexUsed[MAX_PALETTE_PRED_SIZE];
  memset( paletteIndexUsed, 0, sizeof(paletteIndexUsed) );

  Pel *pPred[3] = { pcCU->getLastPaletteInLcuFinal(0), pcCU->getLastPaletteInLcuFinal(1), pcCU->getLastPaletteInLcuFinal(2) };

  UInt scaleX = pcCU->getPic()->getComponentScaleX(COMPONENT_Cb);
  UInt scaleY = pcCU->getPic()->getComponentScaleY(COMPONENT_Cb);

#if !FULL_NBIT
  const BitDepths bitDepths = pcCU->getSlice()->getSPS()->getBitDepths();
#endif
  for( UInt y = 0; y < height; y++ )
  {
    for( UInt x = 0; x < width; x++ )
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);
      UInt paletteIdx = 0;
      UInt minError = MAX_UINT;
      while( paletteIdx < pcCU->getLastPaletteInLcuSizeFinal(0) )
      {
        UInt absError = (abs(pPred[0][paletteIdx] - pSrc[0][pos]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA] - 8));
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          absError += (abs(pPred[1][paletteIdx] - pSrc[1][posC]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8));
          absError += (abs(pPred[2][paletteIdx] - pSrc[2][posC]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8));
        }

        if( absError < minError )
        {
          bestIdx = paletteIdx;
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }

      if( minError <= errorLimit )
      {
        palettePredIndexUsed[bestIdx]++;
      }
    }
  }

  while( idx < maxPaletteSizeSPS )
  {
    UInt maxNoIndexUsed = 0, bestIndex = 0;
    for( UInt i = 0; i < pcCU->getLastPaletteInLcuSizeFinal(0); i++ )
    {
      if( paletteIndexUsed[i] == 0 && palettePredIndexUsed[i] > maxNoIndexUsed )
      {
        maxNoIndexUsed = palettePredIndexUsed[i];
        bestIndex = i;
      }
    }
    if( maxNoIndexUsed > 0 )
    {
      paletteIndexUsed[bestIndex] = 1;

      Palette[0][paletteSize] = pPred[0][bestIndex];
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        Palette[1][paletteSize] = pPred[1][bestIndex];
        Palette[2][paletteSize] = pPred[2][bestIndex];
      }
      paletteSize++;
    }
    else
    {
      break;
    }
    idx++;
  }

  idx = 0;
  for( UInt y = 0; y < height; y++ )
  {
    for( UInt x = 0; x < width; x++ )
    {
      pos = y * width + x;
      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

      UInt paletteIdx = 0;
      UInt minError = MAX_UINT;
      while( paletteIdx < paletteSize )
      {
        UInt absError = (abs(Palette[0][paletteIdx] - pSrc[0][pos]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA] - 8));
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          absError += (abs(Palette[1][paletteIdx] - pSrc[1][posC]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8));
          absError += (abs(Palette[2][paletteIdx] - pSrc[2][posC]) >> DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8));
        }

        if (absError < minError)
        {
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }

      if( minError > errorLimit )
      {
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
        {
          sElement.setAll(pSrc[0][pos], 0, 0);
        }
        else
        {
          sElement.setAll(pSrc[0][pos], pSrc[1][posC], pSrc[2][posC]);
        }
        Int besti = last, bestSAD = (last == -1) ? MAX_UINT : psList[last].getSAD(sElement, pcCU->getSlice()->getSPS()->getBitDepths());
        if (bestSAD)
        {
          for (Int i = idx - 1; i >= 0; i--)
          {
            UInt sad = psList[i].getSAD(sElement, pcCU->getSlice()->getSPS()->getBitDepths());
            if (sad < bestSAD)
            {
              bestSAD = sad;
              besti = i;
              if (!sad)
              {
                break;
              }
            }
          }
        }

        if( besti >= 0 && psList[besti].almostEqualData(sElement, errorLimit, pcCU->getSlice()->getSPS()->getBitDepths()) )
        {
          psList[besti].addElement(sElement);
          last = besti;
        }
        else
        {
          psList[idx].copyDataFrom(sElement);
          psList[idx].cnt = 1;
          last = idx;
          idx++;
        }
      }
    }
  }

  for( Int i = 0; i < maxPaletteSizeSPS; i++ )
  {
    pListSort[i].cnt = 0;
    pListSort[i].setAll(0, 0, 0);
  }

  //bubble sorting
  UInt dictMaxSize = 1;
  for( Int i = 0; i < idx; i++ )
  {
    if( psList[i].cnt > pListSort[dictMaxSize - 1].cnt )
    {
      Int j;
      for( j = dictMaxSize; j > 0; j-- )
      {
        if( psList[i].cnt > pListSort[j - 1].cnt )
        {
          pListSort[j].copyAllFrom(pListSort[j - 1]);
          dictMaxSize = std::min(dictMaxSize + 1, maxPaletteSizeSPS);
        }
        else
        {
          break;
        }
      }
      pListSort[j].copyAllFrom(psList[i]);
    }
  }

  UInt uiNumTotalBits = pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_LUMA) + (pcCU->getSlice()->getSPS()->getBitDepth(CHANNEL_TYPE_CHROMA)<<1);
  Double bitCost = pcCost->getLambda() * uiNumTotalBits;

  for( Int i = 0; i < maxPaletteSizeSPS && paletteSize < maxPaletteSizeSPS; i++ )
  {
    if( pListSort[i].cnt )
    {
      Int half = pListSort[i].cnt >> 1;
      Palette[0][paletteSize] = (pListSort[i].sumData[0] + half) / pListSort[i].cnt;
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        Palette[1][paletteSize] = (pListSort[i].sumData[1] + half) / pListSort[i].cnt;
        Palette[2][paletteSize] = (pListSort[i].sumData[2] + half) / pListSort[i].cnt;
      }

      Bool bDuplicate = false;
      if( pListSort[i].cnt == 1 )
      {
        bDuplicate = true;
      }
      else
      {
        Int best = -1;
        if( errorLimit )
        {
          Double pal[3] = { pListSort[i].sumData[0] / (Double)pListSort[i].cnt, 0, 0 };
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            pal[1] = pListSort[i].sumData[1] / (Double)pListSort[i].cnt;
            pal[2] = pListSort[i].sumData[2] / (Double)pListSort[i].cnt;
          }

          Double err = pal[0] - Palette[0][paletteSize];
          Double bestCost = (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA] - 8)));
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
          {
            err = pal[1] - Palette[1][paletteSize]; bestCost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
            err = pal[2] - Palette[2][paletteSize]; bestCost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
          }
          bestCost = bestCost * pListSort[i].cnt + bitCost;

          for( Int t = 0; t < paletteSize; t++ )
          {
            if( Palette[0][paletteSize] == Palette[0][t] && Palette[1][paletteSize] == Palette[1][t] && Palette[2][paletteSize] == Palette[2][t] )
            {
              bDuplicate = true;
              break;
            }

            err = pal[0] - Palette[0][t];
            Double cost = (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_LUMA] - 8)));
            if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
            {
              err = pal[1] - Palette[1][t]; cost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
              err = pal[2] - Palette[2][t]; cost += (err*err) / (1 << (2 * DISTORTION_PRECISION_ADJUSTMENT(bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8)));
            }
            cost *= pListSort[i].cnt;
            if( cost < bestCost )
            {
              best = t;
              bestCost = cost;
            }
          }
          if( best != -1 )
          {
            bDuplicate = true;
          }
        }
      }

      if( !bDuplicate )
      {
        paletteSize++;
      }
    }
    else
    {
      break;
    }
  }

  UInt palettePredSamples[MAX_PALETTE_SIZE][5];
  memset(palettePredSamples, 0, sizeof(palettePredSamples));
  Int iErrorLimitSqr = 3 * m_paletteErrLimit * m_paletteErrLimit;

  for (UInt y = 0; y < height; y++)
  {
    for (UInt x = 0; x < width; x++)
    {
      Bool discardChroma = y&scaleY || x&scaleX || pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400;
      pos = y * width + x;

      UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

      UInt paletteIdx = 0;
      UInt minError = MAX_UINT;
      while (paletteIdx < paletteSize)
      {
        Int temp=Palette[0][paletteIdx] - pSrc[0][pos];
        UInt absError = (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          temp = Palette[1][paletteIdx] - pSrc[1][posC];
          absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
          temp = Palette[2][paletteIdx] - pSrc[2][posC];
          absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
        }

        if (absError < minError)
        {
          bestIdx = paletteIdx;
          minError = absError;
          if (minError == 0)
          {
            break;
          }
        }
        paletteIdx++;
      }

      UInt escape=0;
      if (minError > iErrorLimitSqr)
      {
        Pel pOrg[3]={ pSrc[0][pos], 0, 0};
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          pOrg[1]=pSrc[1][posC];
          pOrg[2]=pSrc[2][posC];
        }
        UInt errorTemp;
        Double rdCost = xCalcPixelPredRD(pcCU, pOrg, pcCost, &errorTemp);
        if (rdCost<minError)
        {
          escape=1;
        }
      }

      if (escape==0)
      {
        palettePredSamples[bestIdx][0]++;
        palettePredSamples[bestIdx][1] += pSrc[0][pos];
        if (!discardChroma)
        {
          palettePredSamples[bestIdx][2] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[1][posC];
          palettePredSamples[bestIdx][3] += SCM_V0034_PALETTE_CHROMA_SETTINGS*pSrc[2][posC];
          palettePredSamples[bestIdx][4] += SCM_V0034_PALETTE_CHROMA_SETTINGS;
        }
        else if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          palettePredSamples[bestIdx][2] += pSrc[1][posC];
          palettePredSamples[bestIdx][3] += pSrc[2][posC];
          palettePredSamples[bestIdx][4]++;
        }
        m_indexBlock[pos] = bestIdx;
      }
      else
      {
        m_indexBlock[pos]=-1;
      }
    }
  }

  UInt paletteIndBest[MAX_PALETTE_PRED_SIZE];
  UInt paletteSizeTemp=0;
  for (Int i = 0; i < paletteSize; i++)
  {
    if(palettePredSamples[i][0] > 0)
    {
      Palette[0][paletteSizeTemp] = (palettePredSamples[i][1]+palettePredSamples[i][0]/2)/palettePredSamples[i][0];
      if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
      {
        Palette[1][paletteSizeTemp] = (palettePredSamples[i][2] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
        Palette[2][paletteSizeTemp] = (palettePredSamples[i][3] + palettePredSamples[i][4] / 2) / palettePredSamples[i][4];
      }

      Double minError = pcCost->getLambda()*(pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA] + 2 * pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA]);
      Double absError;

      for (UInt y = 0; y < height; y++)
      {
        for (UInt x = 0; x < width; x++)
        {
          pos = y * width + x;
          if (m_indexBlock[pos]==i)
          {
            UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

            Int temp=Palette[0][paletteSizeTemp] - pSrc[0][pos];
            minError += (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
            if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
            {
              temp = Palette[1][paletteSizeTemp] - pSrc[1][posC];
              minError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
              temp = Palette[2][paletteSizeTemp] - pSrc[2][posC];
              minError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
            }
          }
        }
      }

      UInt maxPredCheck=xFindCandidatePalettePredictors(paletteIndBest, pcCU, Palette, pPred, paletteSizeTemp, MAX_PRED_CHEK);
      Int best=-1;
      UInt testedPalettePred;

      for(int t=0; t<maxPredCheck; t++)
      {
        testedPalettePred=paletteIndBest[t];

        absError=0;
        for (UInt y = 0; y < height; y++)
        {
          for (UInt x = 0; x < width; x++)
          {
            pos = y * width + x;
            if (m_indexBlock[pos]==i)
            {
              UInt posC = (y>>scaleY) * (width>>scaleX) + (x>>scaleX);

              Int temp=pPred[0][testedPalettePred] - pSrc[0][pos];
              absError += (( temp * temp ) >>  DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_LUMA]-8) << 1));
              if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
              {
                temp = pPred[1][testedPalettePred] - pSrc[1][posC];
                absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
                temp = pPred[2][testedPalettePred] - pSrc[2][posC];
                absError += ((temp * temp) >> DISTORTION_PRECISION_ADJUSTMENT((pcCU->getSlice()->getSPS()->getBitDepths().recon[CHANNEL_TYPE_CHROMA] - 8) << 1));
              }
            }
          }
          if (absError>minError)
          {
            break;
          }
        }

        if (absError < minError || (absError == minError && best>testedPalettePred))
        {
          best = testedPalettePred;
          minError = absError;
        }
      }

      if( best != -1 )
      {
        Palette[0][paletteSizeTemp] = pPred[0][best];
        if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() != CHROMA_400)
        {
          Palette[1][paletteSizeTemp] = pPred[1][best];
          Palette[2][paletteSizeTemp] = pPred[2][best];
        }
      }


      Bool bDuplicate = false;
      if( palettePredSamples[i][0] == 1 && best == -1 )
      {
        bDuplicate = true;
      }
      else
      {
        for( Int t=0; t<paletteSizeTemp; t++)
        {
          if (pcCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400)
          {
            if (Palette[0][paletteSizeTemp] == Palette[0][t])
            {
              bDuplicate = true;
              break;
            }
          }
          else if( Palette[0][paletteSizeTemp] == Palette[0][t] && Palette[1][paletteSizeTemp] == Palette[1][t] && Palette[2][paletteSizeTemp] == Palette[2][t] )
          {
            bDuplicate = true;
            break;
          }
        }
      }
      if (!bDuplicate)
      {
        paletteSizeTemp++;
      }
    }
  }

  paletteSize=paletteSizeTemp;

  delete[] psList;
  delete[] pListSort;
}

UShort TEncSearch::xGetTruncBinBits( const UInt index, const UInt maxSymbolP1 )
{
  if( maxSymbolP1 < PALETTE_MAX_SYMBOL_P1 )
  {
    return m_truncBinBits[maxSymbolP1][index];
  }
  else
  {
    return xGetTruncatedBinBits(index, maxSymbolP1);
  }
}

UShort TEncSearch::xGetEscapeNumBins( const Pel val )
{
  if( val >= PALETTE_MAX_SYMBOL_P1 )
  {
    return xGetEpExGolombNumBins(val, 3);
  }
  else
  {
    return m_escapeNumBins[val];
  }
}

Void TEncSearch::xInitTBCTable()
{
  assert( PALETTE_MAX_SYMBOL_P1 > 1 );

  m_maxSymbolSize = PALETTE_MAX_SYMBOL_P1;
  m_symbolSize = m_maxSymbolSize - 1;

  m_truncBinBits = new UShort*[m_maxSymbolSize];
  m_truncBinBits[0] = NULL; // not used

  for (UInt i = 1; i < m_maxSymbolSize; i++)
  {
    m_truncBinBits[i] = new UShort[i];
    for (UInt j = 0; j < i; j++)
    {
      m_truncBinBits[i][j] = xGetTruncatedBinBits(j, i);
    }
  }

  m_escapeNumBins = new UShort[m_symbolSize];

  for( UInt i = 0; i < m_symbolSize; i++)
  {
    m_escapeNumBins[i] = xGetEpExGolombNumBins(i, 3);
  }
}

//! \}
