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

/** \file     TEncCu.cpp
    \brief    Coding Unit (CU) encoder class
*/

#include <stdio.h>
#include "TEncTop.h"
#include "TEncCu.h"
#include "TEncAnalyze.h"
#include "TLibCommon/Debug.h"

#include <cmath>
#include <algorithm>
using namespace std;


//! \ingroup TLibEncoder
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

/**
 \param    uhTotalDepth  total number of allowable depth
 \param    uiMaxWidth    largest CU width
 \param    uiMaxHeight   largest CU height
 \param    chromaFormat  chroma format
 */
Void TEncCu::create(UChar uhTotalDepth, UInt uiMaxWidth, UInt uiMaxHeight, ChromaFormat chromaFormat, UInt paletteMaxSize, UInt paletteMaxPredSize )
{
  Int i;

  m_uhTotalDepth   = uhTotalDepth + 1;
  m_ppcBestCU      = new TComDataCU*[m_uhTotalDepth-1];
  m_ppcTempCU      = new TComDataCU*[m_uhTotalDepth-1];

  m_ppcPredYuvBest = new TComYuv*[m_uhTotalDepth-1];
  m_ppcResiYuvBest = new TComYuv*[m_uhTotalDepth-1];
  m_ppcRecoYuvBest = new TComYuv*[m_uhTotalDepth-1];
  m_ppcPredYuvTemp = new TComYuv*[m_uhTotalDepth-1];
  m_ppcResiYuvTemp = new TComYuv*[m_uhTotalDepth-1];
  m_ppcRecoYuvTemp = new TComYuv*[m_uhTotalDepth-1];
  m_ppcOrigYuv     = new TComYuv*[m_uhTotalDepth-1];
  m_ppcNoCorrYuv   = new TComYuv*[m_uhTotalDepth-1];

  UInt uiNumPartitions;
  for( i=0 ; i<m_uhTotalDepth-1 ; i++)
  {
    uiNumPartitions = 1<<( ( m_uhTotalDepth - i - 1 )<<1 );
    UInt uiWidth  = uiMaxWidth  >> i;
    UInt uiHeight = uiMaxHeight >> i;

    m_ppcBestCU[i] = new TComDataCU; m_ppcBestCU[i]->create( chromaFormat, uiNumPartitions, uiWidth, uiHeight, false, uiMaxWidth >> (m_uhTotalDepth - 1), paletteMaxSize, paletteMaxPredSize );
    m_ppcTempCU[i] = new TComDataCU; m_ppcTempCU[i]->create( chromaFormat, uiNumPartitions, uiWidth, uiHeight, false, uiMaxWidth >> (m_uhTotalDepth - 1), paletteMaxSize, paletteMaxPredSize );

    m_ppcPredYuvBest[i] = new TComYuv; m_ppcPredYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);
    m_ppcResiYuvBest[i] = new TComYuv; m_ppcResiYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);
    m_ppcRecoYuvBest[i] = new TComYuv; m_ppcRecoYuvBest[i]->create(uiWidth, uiHeight, chromaFormat);

    m_ppcPredYuvTemp[i] = new TComYuv; m_ppcPredYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);
    m_ppcResiYuvTemp[i] = new TComYuv; m_ppcResiYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);
    m_ppcRecoYuvTemp[i] = new TComYuv; m_ppcRecoYuvTemp[i]->create(uiWidth, uiHeight, chromaFormat);

    m_ppcOrigYuv    [i] = new TComYuv; m_ppcOrigYuv    [i]->create(uiWidth, uiHeight, chromaFormat);
    m_ppcNoCorrYuv  [i] = new TComYuv; m_ppcNoCorrYuv  [i]->create(uiWidth, uiHeight, chromaFormat);
  }

  m_bEncodeDQP                     = false;
  m_stillToCodeChromaQpOffsetFlag  = false;
  m_cuChromaQpOffsetIdxPlus1       = 0;
  m_bFastDeltaQP                   = false;

  // initialize partition order.
  UInt* piTmp = &g_auiZscanToRaster[0];
  initZscanToRaster( m_uhTotalDepth, 1, 0, piTmp);
  initRasterToZscan( uiMaxWidth, uiMaxHeight, m_uhTotalDepth );

  // initialize conversion matrix from partition index to pel
  initRasterToPelXY( uiMaxWidth, uiMaxHeight, m_uhTotalDepth );
}

Void TEncCu::destroy()
{
  Int i;

  for( i=0 ; i<m_uhTotalDepth-1 ; i++)
  {
    if(m_ppcBestCU[i])
    {
      m_ppcBestCU[i]->destroy();      delete m_ppcBestCU[i];      m_ppcBestCU[i] = NULL;
    }
    if(m_ppcTempCU[i])
    {
      m_ppcTempCU[i]->destroy();      delete m_ppcTempCU[i];      m_ppcTempCU[i] = NULL;
    }
    if(m_ppcPredYuvBest[i])
    {
      m_ppcPredYuvBest[i]->destroy(); delete m_ppcPredYuvBest[i]; m_ppcPredYuvBest[i] = NULL;
    }
    if(m_ppcResiYuvBest[i])
    {
      m_ppcResiYuvBest[i]->destroy(); delete m_ppcResiYuvBest[i]; m_ppcResiYuvBest[i] = NULL;
    }
    if(m_ppcRecoYuvBest[i])
    {
      m_ppcRecoYuvBest[i]->destroy(); delete m_ppcRecoYuvBest[i]; m_ppcRecoYuvBest[i] = NULL;
    }
    if(m_ppcPredYuvTemp[i])
    {
      m_ppcPredYuvTemp[i]->destroy(); delete m_ppcPredYuvTemp[i]; m_ppcPredYuvTemp[i] = NULL;
    }
    if(m_ppcResiYuvTemp[i])
    {
      m_ppcResiYuvTemp[i]->destroy(); delete m_ppcResiYuvTemp[i]; m_ppcResiYuvTemp[i] = NULL;
    }
    if(m_ppcRecoYuvTemp[i])
    {
      m_ppcRecoYuvTemp[i]->destroy(); delete m_ppcRecoYuvTemp[i]; m_ppcRecoYuvTemp[i] = NULL;
    }
    if(m_ppcOrigYuv[i])
    {
      m_ppcOrigYuv[i]->destroy();     delete m_ppcOrigYuv[i];     m_ppcOrigYuv[i] = NULL;
    }
    if(m_ppcNoCorrYuv[i])
    {
      m_ppcNoCorrYuv[i]->destroy();   delete m_ppcNoCorrYuv[i];   m_ppcNoCorrYuv[i] = NULL;
    }
  }
  if(m_ppcBestCU)
  {
    delete [] m_ppcBestCU;
    m_ppcBestCU = NULL;
  }
  if(m_ppcTempCU)
  {
    delete [] m_ppcTempCU;
    m_ppcTempCU = NULL;
  }

  if(m_ppcPredYuvBest)
  {
    delete [] m_ppcPredYuvBest;
    m_ppcPredYuvBest = NULL;
  }
  if(m_ppcResiYuvBest)
  {
    delete [] m_ppcResiYuvBest;
    m_ppcResiYuvBest = NULL;
  }
  if(m_ppcRecoYuvBest)
  {
    delete [] m_ppcRecoYuvBest;
    m_ppcRecoYuvBest = NULL;
  }
  if(m_ppcPredYuvTemp)
  {
    delete [] m_ppcPredYuvTemp;
    m_ppcPredYuvTemp = NULL;
  }
  if(m_ppcResiYuvTemp)
  {
    delete [] m_ppcResiYuvTemp;
    m_ppcResiYuvTemp = NULL;
  }
  if(m_ppcRecoYuvTemp)
  {
    delete [] m_ppcRecoYuvTemp;
    m_ppcRecoYuvTemp = NULL;
  }
  if(m_ppcOrigYuv)
  {
    delete [] m_ppcOrigYuv;
    m_ppcOrigYuv = NULL;
  }
  if(m_ppcNoCorrYuv)
  {
    delete [] m_ppcNoCorrYuv;
    m_ppcNoCorrYuv = NULL;
  }
}

/** \param    pcEncTop      pointer of encoder class
 */
Void TEncCu::init( TEncTop* pcEncTop )
{
  m_pcEncCfg           = pcEncTop;
  m_pcPredSearch       = pcEncTop->getPredSearch();
  m_pcTrQuant          = pcEncTop->getTrQuant();
  m_pcRdCost           = pcEncTop->getRdCost();

  m_pcEntropyCoder     = pcEncTop->getEntropyCoder();
  m_pcBinCABAC         = pcEncTop->getBinCABAC();

  m_pppcRDSbacCoder    = pcEncTop->getRDSbacCoder();
  m_pcRDGoOnSbacCoder  = pcEncTop->getRDGoOnSbacCoder();

  m_pcRateCtrl         = pcEncTop->getRateCtrl();
  m_lumaQPOffset       = 0;
  initLumaDeltaQpLUT();
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

/** 
 \param  pCtu pointer of CU data class
 */
Void TEncCu::compressCtu( TComDataCU* pCtu, UChar* lastPaletteSize, Pel lastPalette[][MAX_PALETTE_PRED_SIZE] )
{
  // initialize CU data
  m_ppcBestCU[0]->initCtu( pCtu->getPic(), pCtu->getCtuRsAddr() );
  m_ppcTempCU[0]->initCtu( pCtu->getPic(), pCtu->getCtuRsAddr() );

  for (UChar comp = 0; comp < (pCtu->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++)
  {
    m_ppcBestCU[0]->setLastPaletteInLcuSizeFinal(comp, lastPaletteSize[comp]);
    m_ppcTempCU[0]->setLastPaletteInLcuSizeFinal(comp, lastPaletteSize[comp]);
    for (UInt idx = 0; idx < pCtu->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++)
    {
      m_ppcBestCU[0]->setLastPaletteInLcuFinal(comp, lastPalette[comp][idx], idx);
      m_ppcTempCU[0]->setLastPaletteInLcuFinal(comp, lastPalette[comp][idx], idx);
    }
  }

  // analysis of CU
  DEBUG_STRING_NEW(sDebug)

  xCompressCU( m_ppcBestCU[0], m_ppcTempCU[0], 0 DEBUG_STRING_PASS_INTO(sDebug) );
  DEBUG_STRING_OUTPUT(std::cout, sDebug)

#if ADAPTIVE_QP_SELECTION
  if( m_pcEncCfg->getUseAdaptQpSelect() )
  {
    if(pCtu->getSlice()->getSliceType()!=I_SLICE) //IIII
    {
      xCtuCollectARLStats( pCtu );
    }
  }
#endif
}
/** \param  pCtu  pointer of CU data class
 */
Void TEncCu::encodeCtu ( TComDataCU* pCtu )
{
  if ( pCtu->getSlice()->getPPS()->getUseDQP() )
  {
    setdQPFlag(true);
  }

  if ( pCtu->getSlice()->getUseChromaQpAdj() )
  {
    setCodeChromaQpAdjFlag(true);
  }

  // Encode CU data
  xEncodeCU( pCtu, 0, 0 );
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

Void TEncCu::initLumaDeltaQpLUT()
{
  const LumaLevelToDeltaQPMapping &mapping=m_pcEncCfg->getLumaLevelToDeltaQPMapping();

  if ( !mapping.isEnabled() )
  {
    return;
  }

  // map the sparse LumaLevelToDeltaQPMapping.mapping to a fully populated linear table.

  Int         lastDeltaQPValue=0;
  std::size_t nextSparseIndex=0;
  for(Int index=0; index<LUMA_LEVEL_TO_DQP_LUT_MAXSIZE; index++)
  {
    while (nextSparseIndex < mapping.mapping.size() && index>=mapping.mapping[nextSparseIndex].first)
    {
      lastDeltaQPValue=mapping.mapping[nextSparseIndex].second;
      nextSparseIndex++;
    }
    m_lumaLevelToDeltaQPLUT[index]=lastDeltaQPValue;
  }
}

Int TEncCu::calculateLumaDQP(TComDataCU *pCU, const UInt absPartIdx, const TComYuv * pOrgYuv)
{
  const Pel *pY = pOrgYuv->getAddr(COMPONENT_Y, absPartIdx);
  const Int stride  = pOrgYuv->getStride(COMPONENT_Y);
  Int width = pCU->getWidth(absPartIdx);
  Int height = pCU->getHeight(absPartIdx);
  Double avg = 0;

  // limit the block by picture size
  const TComSPS* pSPS = pCU->getSlice()->getSPS();
  if ( pCU->getCUPelX() + width > pSPS->getPicWidthInLumaSamples() )
  {
    width = pSPS->getPicWidthInLumaSamples() - pCU->getCUPelX();
  }
  if ( pCU->getCUPelY() + height > pSPS->getPicHeightInLumaSamples() )
  {
    height = pSPS->getPicHeightInLumaSamples() - pCU->getCUPelY();
  }

  // Get QP offset derived from Luma level
  if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().mode == LUMALVL_TO_DQP_AVG_METHOD )
  {
    // Use avg method
    Int sum = 0;
    for (Int y = 0; y < height; y++)
    {
      for (Int x = 0; x < width; x++)
      {
        sum += pY[x];
      }
      pY += stride;
    }
    avg = (Double)sum/(width*height);
  }
  else
  {
    // Use maximum luma value
    Int maxVal = 0;
    for (Int y = 0; y < height; y++)
    {
      for (Int x = 0; x < width; x++)
      {
        if (pY[x] > maxVal)
        {
          maxVal = pY[x];
        }
      }
      pY += stride;
    }
    // use a percentage of the maxVal
    avg = (Double)maxVal * m_pcEncCfg->getLumaLevelToDeltaQPMapping().maxMethodWeight;
  }

  Int lumaIdx = Clip3<Int>(0, Int(LUMA_LEVEL_TO_DQP_LUT_MAXSIZE)-1, Int(avg+0.5) );
  Int QP = m_lumaLevelToDeltaQPLUT[lumaIdx];
  return QP;
}

//! Derive small set of test modes for AMP encoder speed-up
#if AMP_ENC_SPEEDUP
#if AMP_MRG
Void TEncCu::deriveTestModeAMP (TComDataCU *pcBestCU, PartSize eParentPartSize, Bool &bTestAMP_Hor, Bool &bTestAMP_Ver, Bool &bTestMergeAMP_Hor, Bool &bTestMergeAMP_Ver)
#else
Void TEncCu::deriveTestModeAMP (TComDataCU *pcBestCU, PartSize eParentPartSize, Bool &bTestAMP_Hor, Bool &bTestAMP_Ver)
#endif
{
  if ( pcBestCU->getPartitionSize(0) == SIZE_2NxN )
  {
    bTestAMP_Hor = true;
  }
  else if ( pcBestCU->getPartitionSize(0) == SIZE_Nx2N )
  {
    bTestAMP_Ver = true;
  }
  else if ( pcBestCU->getPartitionSize(0) == SIZE_2Nx2N && pcBestCU->getMergeFlag(0) == false && pcBestCU->isSkipped(0) == false )
  {
    bTestAMP_Hor = true;
    bTestAMP_Ver = true;
  }

#if AMP_MRG
  //! Utilizing the partition size of parent PU
  if ( eParentPartSize >= SIZE_2NxnU && eParentPartSize <= SIZE_nRx2N )
  {
    bTestMergeAMP_Hor = true;
    bTestMergeAMP_Ver = true;
  }

  if ( eParentPartSize == NUMBER_OF_PART_SIZES ) //! if parent is intra
  {
    if ( pcBestCU->getPartitionSize(0) == SIZE_2NxN )
    {
      bTestMergeAMP_Hor = true;
    }
    else if ( pcBestCU->getPartitionSize(0) == SIZE_Nx2N )
    {
      bTestMergeAMP_Ver = true;
    }
  }

  if ( pcBestCU->getPartitionSize(0) == SIZE_2Nx2N && pcBestCU->isSkipped(0) == false )
  {
    bTestMergeAMP_Hor = true;
    bTestMergeAMP_Ver = true;
  }

  if ( pcBestCU->getWidth(0) == 64 )
  {
    bTestAMP_Hor = false;
    bTestAMP_Ver = false;
  }
#else
  //! Utilizing the partition size of parent PU
  if ( eParentPartSize >= SIZE_2NxnU && eParentPartSize <= SIZE_nRx2N )
  {
    bTestAMP_Hor = true;
    bTestAMP_Ver = true;
  }

  if ( eParentPartSize == SIZE_2Nx2N )
  {
    bTestAMP_Hor = false;
    bTestAMP_Ver = false;
  }
#endif
}
#endif


static Intermediate_Int CalculateMinimumHVLumaActivity(TComDataCU *pcCU, const UInt uiAbsPartIdx, const TComYuv * const * const ppcOrigYuv);

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================
/** Compress a CU block recursively with enabling sub-CTU-level delta QP
 *  - for loop of QP value to compress the current CU with all possible QP
*/
#if AMP_ENC_SPEEDUP
Void TEncCu::xCompressCU( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth DEBUG_STRING_FN_DECLARE(sDebug_), PartSize eParentPartSize )
#else
Void TEncCu::xCompressCU( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, const UInt uiDepth )
#endif
{
  TComMv lastIntraBCMv[2];
  for(Int i=0; i<2; i++)
  {
    lastIntraBCMv[i] = rpcBestCU->getLastIntraBCMv(i);
  }

  UChar lastPaletteSize[3];
  UInt numValidComp = rpcBestCU->getPic()->getNumberValidComponents();
  Pel*  lastPalette[MAX_NUM_COMPONENT];
  for (UInt ch = 0; ch < numValidComp; ch++)
  {
    lastPalette[ch] = (Pel*)xMalloc(Pel, rpcBestCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize());
  }
  for (UInt ch = 0; ch < numValidComp; ch++)
  {
    lastPaletteSize[ch] = rpcBestCU->getLastPaletteInLcuSizeFinal(ch);
    for (UInt i = 0; i < rpcBestCU->getSlice()->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); i++)
    {
      lastPalette[ch][i] = rpcBestCU->getLastPaletteInLcuFinal(ch, i);
    }
  }

  Bool isPerfectMatch = false;
  Bool terminateAllFurtherRDO = false;

  TComMv iMVCandList[4][10];
  memset( iMVCandList, 0, sizeof( TComMv )*4*10 );

  TComPic* pcPic = rpcBestCU->getPic();
  DEBUG_STRING_NEW(sDebug)
  const TComPPS &pps=*(rpcTempCU->getSlice()->getPPS());
  const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
  
  // These are only used if getFastDeltaQp() is true
  const UInt fastDeltaQPCuMaxSize    = Clip3(sps.getMaxCUHeight()>>sps.getLog2DiffMaxMinCodingBlockSize(), sps.getMaxCUHeight(), 32u);

  // get Original YUV data from picture
  m_ppcOrigYuv[uiDepth]->copyFromPicYuv( pcPic->getPicYuvOrg(), rpcBestCU->getCtuRsAddr(), rpcBestCU->getZorderIdxInCtu() );

  // variable for Cbf fast mode PU decision
  Bool    doNotBlockPu = true;
  Bool    earlyDetectionSkipMode = false;

  const UInt uiLPelX   = rpcBestCU->getCUPelX();
  const UInt uiRPelX   = uiLPelX + rpcBestCU->getWidth(0)  - 1;
  const UInt uiTPelY   = rpcBestCU->getCUPelY();
  const UInt uiBPelY   = uiTPelY + rpcBestCU->getHeight(0) - 1;
  const UInt uiWidth   = rpcBestCU->getWidth(0);

  Int iBaseQP = xComputeQP( rpcBestCU, uiDepth );
  Int iMinQP;
  Int iMaxQP;
  Bool isAddLowestQP = false;

  const UInt numberValidComponents = rpcBestCU->getPic()->getNumberValidComponents();

  if( uiDepth <= pps.getMaxCuDQPDepth() )
  {
    Int idQP = m_pcEncCfg->getMaxDeltaQP();
    iMinQP = Clip3( -sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP-idQP );
    iMaxQP = Clip3( -sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP+idQP );
  }
  else
  {
    iMinQP = rpcTempCU->getQP(0);
    iMaxQP = rpcTempCU->getQP(0);
  }

  if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() )
  {
    if ( uiDepth <= pps.getMaxCuDQPDepth() )
    {
      // keep using the same m_QP_LUMA_OFFSET in the same CTU
      m_lumaQPOffset = calculateLumaDQP(rpcTempCU, 0, m_ppcOrigYuv[uiDepth]);
    }
    iMinQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP - m_lumaQPOffset);
    iMaxQP = iMinQP; // force encode choose the modified QO
  }

  if ( m_pcEncCfg->getUseRateCtrl() )
  {
    iMinQP = m_pcRateCtrl->getRCQP();
    iMaxQP = m_pcRateCtrl->getRCQP();
  }

  // transquant-bypass (TQB) processing loop variable initialisation ---

  const Int lowestQP = iMinQP; // For TQB, use this QP which is the lowest non TQB QP tested (rather than QP'=0) - that way delta QPs are smaller, and TQB can be tested at all CU levels.

  if ( (pps.getTransquantBypassEnabledFlag()) )
  {
    isAddLowestQP = true; // mark that the first iteration is to cost TQB mode.
    iMinQP = iMinQP - 1;  // increase loop variable range by 1, to allow testing of TQB mode along with other QPs
    if ( m_pcEncCfg->getCUTransquantBypassFlagForceValue() )
    {
      iMaxQP = iMinQP;
    }
  }

  TComSlice * pcSlice = rpcTempCU->getPic()->getSlice(rpcTempCU->getPic()->getCurrSliceIdx());

  if(rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans())
  {
    m_ppcBestCU[uiDepth]->setTmpIntraBCRDCost(MAX_DOUBLE);
    m_ppcTempCU[uiDepth]->setTmpIntraBCRDCost(MAX_DOUBLE);
    m_ppcBestCU[uiDepth]->setIntraBCCSCEnabled(true);
    m_ppcTempCU[uiDepth]->setIntraBCCSCEnabled(true);

    m_ppcBestCU[uiDepth]->setTmpInterRDCost(MAX_DOUBLE);
    m_ppcTempCU[uiDepth]->setTmpInterRDCost(MAX_DOUBLE);
    m_ppcBestCU[uiDepth]->setInterCSCEnabled(true);
    m_ppcTempCU[uiDepth]->setInterCSCEnabled(true);
    setEnableIntraTUACT(uiDepth, pcSlice);
    setEnableIBCTUACT(uiDepth, pcSlice);
    setEnableInterTUACT(uiDepth, pcSlice);
  }

  const Bool bBoundary = !( uiRPelX < sps.getPicWidthInLumaSamples() && uiBPelY < sps.getPicHeightInLumaSamples() );

  if ( !bBoundary )
  {
    for (Int iQP=iMinQP; iQP<=iMaxQP; iQP++)
    {
      const Bool bIsLosslessMode = isAddLowestQP && (iQP == iMinQP);

      if (bIsLosslessMode)
      {
        iQP = lowestQP;
      }
      if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() && uiDepth <= pps.getMaxCuDQPDepth() )
      {
        getSliceEncoder()->updateLambda(pcSlice, iQP);
      }

      m_cuChromaQpOffsetIdxPlus1 = 0;
      if (pcSlice->getUseChromaQpAdj())
      {
        /* Pre-estimation of chroma QP based on input block activity may be performed
         * here, using for example m_ppcOrigYuv[uiDepth] */
        /* To exercise the current code, the index used for adjustment is based on
         * block position
         */
        Int lgMinCuSize = sps.getLog2MinCodingBlockSize() +
                          std::max<Int>(0, sps.getLog2DiffMaxMinCodingBlockSize()-Int(pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth()));
        m_cuChromaQpOffsetIdxPlus1 = ((uiLPelX >> lgMinCuSize) + (uiTPelY >> lgMinCuSize)) % (pps.getPpsRangeExtension().getChromaQpOffsetListLen() + 1);
      }

      rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

      // do inter modes, SKIP and 2Nx2N
      if ( ( !rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->getSliceType() != I_SLICE ) ||
           ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && !rpcBestCU->getSlice()->isOnlyCurrentPictureAsReference() ) )
      {
        if ( m_pcEncCfg->getUseHashBasedME() )
        {
          xCheckRDCostHashInter( rpcBestCU, rpcTempCU, isPerfectMatch DEBUG_STRING_PASS_INTO(sDebug) );
          rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

          if ( isPerfectMatch )
          {
            if ( uiDepth == 0 )
            {
              terminateAllFurtherRDO = true;
            }
          }
        }

        // 2Nx2N
        if(m_pcEncCfg->getUseEarlySkipDetection() && !terminateAllFurtherRDO)
        {
          xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug) );
          rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );//by Competition for inter_2Nx2N
        }
        // SKIP
        xCheckRDCostMerge2Nx2N( rpcBestCU, rpcTempCU DEBUG_STRING_PASS_INTO(sDebug), &earlyDetectionSkipMode, terminateAllFurtherRDO );//by Merge for inter_2Nx2N
        rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

        if(!m_pcEncCfg->getUseEarlySkipDetection() && !terminateAllFurtherRDO)
        {
          // 2Nx2N, NxN
          xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug) );
          rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
          if(m_pcEncCfg->getUseCbfFastMode())
          {
            doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
          }
        }
      }

      if (bIsLosslessMode) // Restore loop variable if lossless mode was searched.
      {
        iQP = iMinQP;
      }
    }

    if(!earlyDetectionSkipMode && !terminateAllFurtherRDO)
    {
      for (Int iQP=iMinQP; iQP<=iMaxQP; iQP++)
      {
        const Bool bIsLosslessMode = isAddLowestQP && (iQP == iMinQP); // If lossless, then iQP is irrelevant for subsequent modules.

        if (bIsLosslessMode)
        {
          iQP = lowestQP;
        }

        rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

        // do inter modes, NxN, 2NxN, and Nx2N
        if ( ( !rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->getSliceType() != I_SLICE ) ||
             ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && !rpcBestCU->getSlice()->isOnlyCurrentPictureAsReference() ) )
        {
          // 2Nx2N, NxN

          if(!( (rpcBestCU->getWidth(0)==8) && (rpcBestCU->getHeight(0)==8) ))
          {
            if( uiDepth == sps.getLog2DiffMaxMinCodingBlockSize() && doNotBlockPu)
            {
              xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug)   );
              rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            }
          }

          if(doNotBlockPu)
          {
            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_Nx2N DEBUG_STRING_PASS_INTO(sDebug), false, iMVCandList[SIZE_Nx2N] );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_Nx2N )
            {
              doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
            }
          }
          if(doNotBlockPu)
          {
            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxN DEBUG_STRING_PASS_INTO(sDebug), false, iMVCandList[SIZE_2NxN] );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxN)
            {
              doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
            }
          }

          //! Try AMP (SIZE_2NxnU, SIZE_2NxnD, SIZE_nLx2N, SIZE_nRx2N)
          if(sps.getUseAMP() && uiDepth < sps.getLog2DiffMaxMinCodingBlockSize() )
          {
#if AMP_ENC_SPEEDUP
            Bool bTestAMP_Hor = false, bTestAMP_Ver = false;

#if AMP_MRG
            Bool bTestMergeAMP_Hor = false, bTestMergeAMP_Ver = false;

            deriveTestModeAMP (rpcBestCU, eParentPartSize, bTestAMP_Hor, bTestAMP_Ver, bTestMergeAMP_Hor, bTestMergeAMP_Ver);
#else
            deriveTestModeAMP (rpcBestCU, eParentPartSize, bTestAMP_Hor, bTestAMP_Ver);
#endif

            //! Do horizontal AMP
            if ( bTestAMP_Hor )
            {
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnU DEBUG_STRING_PASS_INTO(sDebug) );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnU )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnD DEBUG_STRING_PASS_INTO(sDebug) );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnD )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
            }
#if AMP_MRG
            else if ( bTestMergeAMP_Hor )
            {
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnU DEBUG_STRING_PASS_INTO(sDebug), true );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnU )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnD DEBUG_STRING_PASS_INTO(sDebug), true );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_2NxnD )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
            }
#endif

            //! Do horizontal AMP
            if ( bTestAMP_Ver )
            {
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nLx2N DEBUG_STRING_PASS_INTO(sDebug) );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_nLx2N )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nRx2N DEBUG_STRING_PASS_INTO(sDebug) );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
              }
            }
#if AMP_MRG
            else if ( bTestMergeAMP_Ver )
            {
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nLx2N DEBUG_STRING_PASS_INTO(sDebug), true );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                if(m_pcEncCfg->getUseCbfFastMode() && rpcBestCU->getPartitionSize(0) == SIZE_nLx2N )
                {
                  doNotBlockPu = rpcBestCU->getQtRootCbf( 0 ) != 0;
                }
              }
              if(doNotBlockPu)
              {
                xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nRx2N DEBUG_STRING_PASS_INTO(sDebug), true );
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
              }
            }
#endif

#else
            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnU );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_2NxnD );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nLx2N );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

            xCheckRDCostInter( rpcBestCU, rpcTempCU, SIZE_nRx2N );
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

#endif
          }
        }

        // do normal intra modes
        // speedup for inter frames
        Double intraCost = MAX_DOUBLE;
        Double dIntraBcCostPred = 0.0;
        if ( ( !rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->getSliceType() == I_SLICE ) ||
             ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->isOnlyCurrentPictureAsReference() ) ||
             !rpcBestCU->isSkipped(0) ) // avoid very complex intra if it is unlikely
        {
          if (m_pcEncCfg->getUseIntraBlockCopyFastSearch() && rpcTempCU->getWidth(0) <= SCM_S0067_MAX_CAND_SIZE )
          {
            xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, false, SIZE_2Nx2N, dIntraBcCostPred, true DEBUG_STRING_PASS_INTO(sDebug));
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
          }

#if MCTS_ENC_CHECK
          if ( m_pcEncCfg->getTMCTSSEITileConstraint() || (rpcBestCU->getPredictionMode(0) == NUMBER_OF_PREDICTION_MODES) ||
              ((!m_pcEncCfg->getDisableIntraPUsInInterSlices()) && (
              (rpcBestCU->getCbf(0, COMPONENT_Y) != 0) ||
              ((rpcBestCU->getCbf(0, COMPONENT_Cb) != 0) && (numberValidComponents > COMPONENT_Cb)) ||
              ((rpcBestCU->getCbf(0, COMPONENT_Cr) != 0) && (numberValidComponents > COMPONENT_Cr))  // avoid very complex intra if it is unlikely
              )))
          {
#else
          if( rpcBestCU->getPredictionMode(0) == NUMBER_OF_PREDICTION_MODES ||
              ((!m_pcEncCfg->getDisableIntraPUsInInterSlices()) && (
               (rpcBestCU->getCbf( 0, COMPONENT_Y  ) != 0)                                            ||
              ((rpcBestCU->getCbf( 0, COMPONENT_Cb ) != 0) && (numberValidComponents > COMPONENT_Cb)) ||
              ((rpcBestCU->getCbf( 0, COMPONENT_Cr ) != 0) && (numberValidComponents > COMPONENT_Cr))  ) // avoid very complex intra if it is unlikely
            ))
          {
#endif 
            if(rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && ( !bIsLosslessMode || (sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA ))))
            {
              Double tempIntraCost = MAX_DOUBLE;
              TComDataCU *pStoredBestCU = rpcBestCU;
              if(m_pcEncCfg->getRGBFormatFlag())
              {
                xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tempIntraCost, SIZE_2Nx2N, ACT_TRAN_CLR, false DEBUG_STRING_PASS_INTO(sDebug) );
              }
              else
              {
                if( uiDepth < 3 || bIsLosslessMode )
                {
                  xCheckRDCostIntra( rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug));
                }
                else
                {
                  xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N, ACT_ORG_CLR, false DEBUG_STRING_PASS_INTO(sDebug) );
                }
              }
              Bool bSkipRGBCoding = pStoredBestCU == rpcBestCU ? !rpcTempCU->getQtRootCbf( 0 ): !rpcBestCU->getQtRootCbf( 0 );
              if(!bSkipRGBCoding)
              {
                rpcTempCU->initRQTData( uiDepth, rpcBestCU, (pStoredBestCU != rpcBestCU), false, true );
                if(m_pcEncCfg->getRGBFormatFlag())
                {
                  if( !getEnableIntraTUACT() )
                  {
                    xCheckRDCostIntra( rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug), true );
                  }
                  else
                  {
                    xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N, ACT_TWO_CLR, true DEBUG_STRING_PASS_INTO(sDebug) );
                  }
                }
                else
                {
                  xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tempIntraCost, SIZE_2Nx2N, ACT_TRAN_CLR, true DEBUG_STRING_PASS_INTO(sDebug) );
                }
                intraCost = std::min(intraCost, tempIntraCost);
              }
              else
                intraCost = tempIntraCost;
            }
            else
            {
              xCheckRDCostIntra( rpcBestCU, rpcTempCU, intraCost, SIZE_2Nx2N DEBUG_STRING_PASS_INTO(sDebug) );
            }
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            if( uiDepth == sps.getLog2DiffMaxMinCodingBlockSize() )
            {
              if( rpcTempCU->getWidth(0) > ( 1 << sps.getQuadtreeTULog2MinSize() ) )
              {
                Double tmpIntraCost;
                if(rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && ( !bIsLosslessMode || (sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA ))))
                {
                  Double      tempIntraCost = MAX_DOUBLE;
                  TComDataCU *pStoredBestCU = rpcBestCU;
                  if(m_pcEncCfg->getRGBFormatFlag())
                  {
                    xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tempIntraCost, SIZE_NxN, ACT_TRAN_CLR, false DEBUG_STRING_PASS_INTO(sDebug) );
                  }
                  else
                  {
                    if( uiDepth < 3 || bIsLosslessMode )
                    {
                      xCheckRDCostIntra( rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug) );
                    }
                    else
                    {
                      xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN, ACT_ORG_CLR, false DEBUG_STRING_PASS_INTO(sDebug) );
                    }
                  }
                  Bool bSkipRGBCoding = pStoredBestCU == rpcBestCU ? !rpcTempCU->getQtRootCbf( 0 ) : !rpcBestCU->getQtRootCbf( 0 );
                  if(!bSkipRGBCoding)
                  {
                    rpcTempCU->initRQTData( uiDepth, rpcBestCU, (pStoredBestCU != rpcBestCU), false, true );
                    if(m_pcEncCfg->getRGBFormatFlag())
                    {
                      if( !getEnableIntraTUACT() )
                      {
                        xCheckRDCostIntra( rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug), true );
                      }
                      else
                      {
                        xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN, ACT_TWO_CLR, true DEBUG_STRING_PASS_INTO(sDebug) );
                      }
                    }
                    else
                    {
                      xCheckRDCostIntraCSC( rpcBestCU, rpcTempCU, tempIntraCost, SIZE_NxN, ACT_TRAN_CLR, true DEBUG_STRING_PASS_INTO(sDebug) );
                    }
                    tmpIntraCost = std::min(tmpIntraCost, tempIntraCost);
                  }
                  else
                    tmpIntraCost = tempIntraCost;
                }
                else
                {
                  xCheckRDCostIntra( rpcBestCU, rpcTempCU, tmpIntraCost, SIZE_NxN DEBUG_STRING_PASS_INTO(sDebug)   );
                }
                intraCost = std::min(intraCost, tmpIntraCost);
                rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
              }
            }
          }

          if( rpcBestCU->isIntraBC( 0 ) )
          {
            intraCost = dIntraBcCostPred;
          }
        }

        intraCost = MAX_DOUBLE;

        // test PCM
        if(sps.getUsePCM()
          && rpcTempCU->getWidth(0) <= (1<<sps.getPCMLog2MaxSize())
          && rpcTempCU->getWidth(0) >= (1<<sps.getPCMLog2MinSize()) )
        {
          UInt uiRawBits = getTotalBits(rpcBestCU->getWidth(0), rpcBestCU->getHeight(0), rpcBestCU->getPic()->getChromaFormat(), sps.getBitDepths().recon);
          UInt uiBestBits = rpcBestCU->getTotalBits();
          if((uiBestBits > uiRawBits) || (rpcBestCU->getTotalCost() > m_pcRdCost->calcRdCost(uiRawBits, 0)))
          {
            xCheckIntraPCM (rpcBestCU, rpcTempCU);
            rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
          }
        }

        Bool bUse1DSearchFor8x8        = false;
        Bool bSkipIntraBlockCopySearch = false;

        if (rpcTempCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy())
        {
          xCheckRDCostIntraBCMerge2Nx2N( rpcBestCU, rpcTempCU );
          rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
        }

        if( !rpcBestCU->isSkipped(0) ) // avoid very complex intra if it is unlikely
        {
          if (m_pcEncCfg->getUseIntraBlockCopyFastSearch())
          {
            bSkipIntraBlockCopySearch = ((rpcTempCU->getWidth(0) > 16) || (intraCost < std::max(32*m_pcRdCost->getLambda(), 48.0)));

            if (rpcTempCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() &&
                !bSkipIntraBlockCopySearch &&
                rpcTempCU->getWidth(0) == 8 &&
                !m_ppcBestCU[uiDepth -1]->isIntraBC(0) )
            {
              bUse1DSearchFor8x8 = (CalculateMinimumHVLumaActivity(rpcTempCU, 0, m_ppcOrigYuv) < (168 << (sps.getBitDepth( CHANNEL_TYPE_LUMA ) - 8)));
            }
          }

          if (rpcTempCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy())
          {
            if (!bSkipIntraBlockCopySearch)
            {
              Double adIntraBcCost[NUMBER_OF_PART_SIZES] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
              xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, bUse1DSearchFor8x8, SIZE_2Nx2N, adIntraBcCost[SIZE_2Nx2N] DEBUG_STRING_PASS_INTO(sDebug));
              rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

              if (m_pcEncCfg->getUseIntraBlockCopyFastSearch())
              {
                if( uiDepth == sps.getLog2DiffMaxMinCodingBlockSize() ) // Only additionally check Nx2N, 2NxN, NxN at the bottom level in fast search
                {
                  intraCost = std::min( intraCost, adIntraBcCost[SIZE_2Nx2N] );

                  Double dTH2 = std::max( 60 * m_pcRdCost->getLambda(),  56.0 );
                  Double dTH3 = std::max( 66 * m_pcRdCost->getLambda(), 800.0 );
                  if( intraCost >= dTH2 ) // only check Nx2N depending on best intraCost (and intraBCcost so far)
                  {
                    xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, true, SIZE_Nx2N, adIntraBcCost[SIZE_Nx2N], false, (iMVCandList[SIZE_Nx2N]+8) DEBUG_STRING_PASS_INTO(sDebug));
                    rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                    intraCost = std::min( intraCost, adIntraBcCost[SIZE_Nx2N] );

                    if ( ( !rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->getSliceType() != I_SLICE ) ||
                         ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && !rpcBestCU->getSlice()->isOnlyCurrentPictureAsReference() ) )
                    {
                      xCheckRDCostIntraBCMixed( rpcBestCU, rpcTempCU, SIZE_Nx2N, adIntraBcCost[SIZE_Nx2N] DEBUG_STRING_PASS_INTO(sDebug), iMVCandList[SIZE_Nx2N]);
                      rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                      intraCost = std::min( intraCost, adIntraBcCost[SIZE_Nx2N] );
                    }
                  }

                  if( intraCost >= dTH2 && !bIsLosslessMode ) // only check 2NxN depending on best intraCost (and intraBCcost so far) and if it is not lossless
                  {
                    xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, ( bUse1DSearchFor8x8 || intraCost < dTH3 ), SIZE_2NxN, adIntraBcCost[SIZE_2NxN], false, (iMVCandList[SIZE_2NxN]+8) DEBUG_STRING_PASS_INTO(sDebug));
                    rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                    intraCost = std::min( intraCost, adIntraBcCost[SIZE_2NxN] );

                    if ( ( !rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && rpcBestCU->getSlice()->getSliceType() != I_SLICE ) ||
                         ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseIntraBlockCopy() && !rpcBestCU->getSlice()->isOnlyCurrentPictureAsReference() ) )
                    {
                      xCheckRDCostIntraBCMixed( rpcBestCU, rpcTempCU, SIZE_2NxN, adIntraBcCost[SIZE_2NxN] DEBUG_STRING_PASS_INTO(sDebug), iMVCandList[SIZE_2NxN]);
                      rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                      intraCost = std::min( intraCost, adIntraBcCost[SIZE_2NxN] );
                    }
                }
              }
            }
            else
            {
              // full search (bUse1DSearchFor8x8 will be false but is kept here for consistency).

              xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, bUse1DSearchFor8x8, SIZE_Nx2N, adIntraBcCost[SIZE_Nx2N] DEBUG_STRING_PASS_INTO(sDebug));
              rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
              xCheckRDCostIntraBC( rpcBestCU, rpcTempCU, bUse1DSearchFor8x8, SIZE_2NxN, adIntraBcCost[SIZE_2NxN] DEBUG_STRING_PASS_INTO(sDebug));
              rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
            }
          }
        }

          if (bIsLosslessMode) // Restore loop variable if lossless mode was searched.
          {
            iQP = iMinQP;
          }
          if ( rpcBestCU->getSlice()->getSPS()->getSpsScreenExtension().getUsePaletteMode() )
          {
            //Change Palette QP dependent error limit
            Int iQP_Palette=Int(rpcBestCU->getQP(0));
            Int iQPrem = iQP_Palette % 6;
            Int iQPper = iQP_Palette / 6;
            Double quantiserScale = g_quantScales[iQPrem];
            Int quantiserRightShift = QUANT_SHIFT + iQPper;

            Double dQP=((Double)(1<<quantiserRightShift))/quantiserScale;

            UInt paletteQP;
            paletteQP=(UInt)(2.0*dQP/3.0+0.5);
            m_pcPredSearch->setPaletteErrLimit(paletteQP);

            Bool forcePalettePrediction = false;
            for( UChar ch = 0; ch < numValidComp; ch++ )
            {
              forcePalettePrediction = forcePalettePrediction || ( rpcTempCU->getLastPaletteInLcuSizeFinal( ch ) > 0 );
            }

            UInt iterNumber=0, paletteSize[2] = {MAX_PALETTE_SIZE, MAX_PALETTE_SIZE}, testedModes[4];

            if( rpcTempCU->getWidth(0) != 64)
            {
              iterNumber = 0;
              testedModes[iterNumber]=xCheckPaletteMode( rpcBestCU, rpcTempCU, false, iterNumber, paletteSize);
              rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

              if( !bIsLosslessMode )
              {
                if (paletteSize[0]>2 && testedModes[0]>0)
                {
                  iterNumber = 2;
                  testedModes[iterNumber]=xCheckPaletteMode( rpcBestCU, rpcTempCU, false, iterNumber, paletteSize);
                  rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                }

                if( forcePalettePrediction)
                {
                  iterNumber = 1;
                  testedModes[iterNumber]=xCheckPaletteMode( rpcBestCU, rpcTempCU, true, iterNumber, paletteSize+1);
                  rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                }

                if (forcePalettePrediction && paletteSize[1]>2 && testedModes[1]>0)
                {
                  iterNumber = 3;
                  testedModes[iterNumber]=xCheckPaletteMode( rpcBestCU, rpcTempCU, true, iterNumber, paletteSize+1);
                  rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
                }
              }
            }
          }
        }
      }
    }

    // If Intra BC keep last coded Mv
    if(rpcBestCU->isInter(0))
    {
      Int iRefIdxFirst = rpcBestCU->getCUMvField( REF_PIC_LIST_0 )->getRefIdx( 0 );
      Int iRefIdxLast = rpcBestCU->getCUMvField( REF_PIC_LIST_0 )->getRefIdx( rpcBestCU->getTotalNumPart() - 1 );
      Bool isIntraBCFirst = ( iRefIdxFirst >= 0 ) ? rpcBestCU->getSlice()->getRefPic( REF_PIC_LIST_0, iRefIdxFirst )->getPOC() == rpcBestCU->getSlice()->getPOC() : false;
      Bool isIntraBCLast = ( iRefIdxLast >= 0 ) ? rpcBestCU->getSlice()->getRefPic( REF_PIC_LIST_0, iRefIdxLast )->getPOC() == rpcBestCU->getSlice()->getPOC() : false;

      if (  isIntraBCFirst || isIntraBCLast  )
      {
        if( rpcBestCU->getPartitionSize( 0 ) == SIZE_2Nx2N)
        {
          if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) != rpcBestCU->getLastIntraBCMv(0))
          {
            rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(0), 1 );
            rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) );
          }
        }
        else if(rpcBestCU->getPartitionSize( 0 ) == SIZE_2NxN || rpcBestCU->getPartitionSize( 0 ) == SIZE_Nx2N)
        {
          // mixed PU, only one partition is IntraBC coded
          if( isIntraBCFirst != isIntraBCLast )
          {
            if(isIntraBCFirst)
            {
              // Part 0
              if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ) != rpcBestCU->getLastIntraBCMv())
              {
                rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(), 1);
                rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ) );
              }
            }
            else if(isIntraBCLast)
            {
              // Part 1
              if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) != rpcBestCU->getLastIntraBCMv())
              {
                rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(), 1);
                rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) );
              }
            }
          }
          else // normal IntraBC CU
          {
            // Part 0
            if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ) != rpcBestCU->getLastIntraBCMv())
            {
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(), 1);
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( 0 ) );
            }
            // Part 1
            if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) != rpcBestCU->getLastIntraBCMv())
            {
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(), 1);
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 1 ) );
            }
          }
        }
        else
        {
          // NxN
          for(Int part=0; part<4; part++)
          {
            if( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 4 + part ) != rpcBestCU->getLastIntraBCMv())
            {
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getLastIntraBCMv(), 1 );
              rpcBestCU->setLastIntraBCMv( rpcBestCU->getCUMvField(REF_PIC_LIST_0)->getMv( rpcBestCU->getTotalNumPart() - 4 + part ) );
            }
          }
        }
      }
    } // is inter
    if (rpcBestCU->getPaletteModeFlag(0))
    {
      rpcBestCU->saveLastPaletteInLcuFinal( rpcBestCU, 0, rpcBestCU->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3 );
    }

    if( rpcBestCU->getTotalCost()!=MAX_DOUBLE )
    {
      m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]);
      m_pcEntropyCoder->resetBits();
      m_pcEntropyCoder->encodeSplitFlag( rpcBestCU, 0, uiDepth, true );
      rpcBestCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // split bits
      rpcBestCU->getTotalBins() += ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
      rpcBestCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcBestCU->getTotalBits(), rpcBestCU->getTotalDistortion() );
      m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]);
    }
  }

  // copy original YUV samples to PCM buffer
  if( rpcBestCU->getTotalCost()!=MAX_DOUBLE && rpcBestCU->isLosslessCoded(0) && (rpcBestCU->getIPCMFlag(0) == false))
  {
    xFillPCMBuffer(rpcBestCU, m_ppcOrigYuv[uiDepth]);
  }

  if( uiDepth == pps.getMaxCuDQPDepth() )
  {
    Int idQP = m_pcEncCfg->getMaxDeltaQP();
    iMinQP = Clip3( -sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP-idQP );
    iMaxQP = Clip3( -sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP+idQP );
  }
  else if( uiDepth < pps.getMaxCuDQPDepth() )
  {
    iMinQP = iBaseQP;
    iMaxQP = iBaseQP;
  }
  else
  {
    const Int iStartQP = rpcTempCU->getQP(0);
    iMinQP = iStartQP;
    iMaxQP = iStartQP;
  }

  if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() )
  {
    iMinQP = Clip3(-sps.getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQP - m_lumaQPOffset);
    iMaxQP = iMinQP;
  }

  if ( m_pcEncCfg->getUseRateCtrl() )
  {
    iMinQP = m_pcRateCtrl->getRCQP();
    iMaxQP = m_pcRateCtrl->getRCQP();
  }

  if ( m_pcEncCfg->getCUTransquantBypassFlagForceValue() )
  {
    iMaxQP = iMinQP; // If all TUs are forced into using transquant bypass, do not loop here.
  }

  Bool bSubBranch = bBoundary || !( m_pcEncCfg->getUseEarlyCU() && rpcBestCU->getTotalCost()!=MAX_DOUBLE && rpcBestCU->isSkipped(0) );

  if( rpcBestCU->isIntraBC(0) && rpcBestCU->getQtRootCbf(0) == 0 )
  {
    bSubBranch = false;
  }

  if( !terminateAllFurtherRDO && bSubBranch && uiDepth < sps.getLog2DiffMaxMinCodingBlockSize() && (!getFastDeltaQp() || uiWidth > fastDeltaQPCuMaxSize || bBoundary))
  {
    PaletteInfoBuffer tempPalettePredictor;

    if( iMinQP != iMaxQP )
    {
      memcpy( tempPalettePredictor.lastPaletteSize, lastPaletteSize, sizeof( lastPaletteSize ) );
      memcpy( tempPalettePredictor.lastPalette[0],  lastPalette[0], sizeof( tempPalettePredictor.lastPalette[0] ) );
      memcpy( tempPalettePredictor.lastPalette[1],  lastPalette[1], sizeof( tempPalettePredictor.lastPalette[1] ) );
      memcpy( tempPalettePredictor.lastPalette[2],  lastPalette[2], sizeof( tempPalettePredictor.lastPalette[2] ) );
    }

    // further split
    Double splitTotalCost = 0;

    for (Int iQP=iMinQP; iQP<=iMaxQP; iQP++)
    {
      const Bool bIsLosslessMode = false; // False at this level. Next level down may set it to true.

      rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );

      UChar       uhNextDepth         = uiDepth+1;
      TComDataCU* pcSubBestPartCU     = m_ppcBestCU[uhNextDepth];
      TComDataCU* pcSubTempPartCU     = m_ppcTempCU[uhNextDepth];
      DEBUG_STRING_NEW(sTempDebug)

      if( iMinQP != iMaxQP && iQP != iMinQP )
      {
        memcpy( lastPaletteSize, tempPalettePredictor.lastPaletteSize, sizeof( lastPaletteSize ) );
        memcpy( lastPalette[0],  tempPalettePredictor.lastPalette[0],  sizeof( tempPalettePredictor.lastPalette[0] ) );
        memcpy( lastPalette[1],  tempPalettePredictor.lastPalette[1],  sizeof( tempPalettePredictor.lastPalette[1] ) );
        memcpy( lastPalette[2],  tempPalettePredictor.lastPalette[2],  sizeof( tempPalettePredictor.lastPalette[2] ) );
      }

      for ( UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++ )
      {
        pcSubBestPartCU->initSubCU( rpcTempCU, uiPartUnitIdx, uhNextDepth, iQP );           // clear sub partition datas or init.
        pcSubTempPartCU->initSubCU( rpcTempCU, uiPartUnitIdx, uhNextDepth, iQP );           // clear sub partition datas or init.

        for(Int i=0; i<2; i++)
        {
          pcSubBestPartCU->setLastIntraBCMv( lastIntraBCMv[i], i );
          pcSubTempPartCU->setLastIntraBCMv( lastIntraBCMv[i], i );
        }

        for (UInt ch = 0; ch < numValidComp; ch++)
        {
          pcSubBestPartCU->setLastPaletteInLcuSizeFinal(ch, lastPaletteSize[ch]);
          pcSubTempPartCU->setLastPaletteInLcuSizeFinal(ch, lastPaletteSize[ch]);
          for (UInt i = 0; i < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); i++)
          {
            pcSubBestPartCU->setLastPaletteInLcuFinal(ch, lastPalette[ch][i], i);
            pcSubTempPartCU->setLastPaletteInLcuFinal(ch, lastPalette[ch][i], i);
          }
        }

        if( ( pcSubBestPartCU->getCUPelX() < sps.getPicWidthInLumaSamples() ) && ( pcSubBestPartCU->getCUPelY() < sps.getPicHeightInLumaSamples() ) )
        {
          if ( 0 == uiPartUnitIdx) //initialize RD with previous depth buffer
          {
            m_pppcRDSbacCoder[uhNextDepth][CI_CURR_BEST]->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);
          }
          else
          {
            m_pppcRDSbacCoder[uhNextDepth][CI_CURR_BEST]->load(m_pppcRDSbacCoder[uhNextDepth][CI_NEXT_BEST]);
          }

#if AMP_ENC_SPEEDUP
          DEBUG_STRING_NEW(sChild)
          if ( !(rpcBestCU->getTotalCost()!=MAX_DOUBLE && rpcBestCU->isInter(0)) )
          {
            xCompressCU( pcSubBestPartCU, pcSubTempPartCU, uhNextDepth DEBUG_STRING_PASS_INTO(sChild), NUMBER_OF_PART_SIZES );
          }
          else
          {

            xCompressCU( pcSubBestPartCU, pcSubTempPartCU, uhNextDepth DEBUG_STRING_PASS_INTO(sChild), rpcBestCU->getPartitionSize(0) );
          }
          DEBUG_STRING_APPEND(sTempDebug, sChild)
#else
          xCompressCU( pcSubBestPartCU, pcSubTempPartCU, uhNextDepth );
#endif

          // NOTE: (0,0) is used as an indicator that IntraBC has not been used within the CU.
          if( pcSubBestPartCU->getLastIntraBCMv().getHor() != 0 || pcSubBestPartCU->getLastIntraBCMv().getVer() != 0 )
          {
            for(Int i=0; i<2; i++)
            {
              lastIntraBCMv[i] = pcSubBestPartCU->getLastIntraBCMv(i);
            }
          }

          if (pcSubBestPartCU->getLastPaletteInLcuSizeFinal(COMPONENT_Y) != 0)
          {
            for (UInt ch = 0; ch < numValidComp; ch++)
            {
              lastPaletteSize[ch] = pcSubBestPartCU->getLastPaletteInLcuSizeFinal(ch);
              for (UInt i = 0; i < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); i++)
              {

                lastPalette[ch][i] = pcSubBestPartCU->getLastPaletteInLcuFinal(ch, i);
              }
            }
          }

          rpcTempCU->copyPartFrom( pcSubBestPartCU, uiPartUnitIdx, uhNextDepth );         // Keep best part data to current temporary data.
          xCopyYuv2Tmp( pcSubBestPartCU->getTotalNumPart()*uiPartUnitIdx, uhNextDepth );
          if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() && pps.getMaxCuDQPDepth() >= 1 )
          {
            splitTotalCost += pcSubBestPartCU->getTotalCost();
          }
        }
        else
        {
          pcSubBestPartCU->copyToPic( uhNextDepth );
          rpcTempCU->copyPartFrom( pcSubBestPartCU, uiPartUnitIdx, uhNextDepth );
        }
      }

      m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uhNextDepth][CI_NEXT_BEST]);
      if( !bBoundary )
      {
        m_pcEntropyCoder->resetBits();
        m_pcEntropyCoder->encodeSplitFlag( rpcTempCU, 0, uiDepth, true );
        if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() && pps.getMaxCuDQPDepth() >= 1 )
        {
          Int splitBits = m_pcEntropyCoder->getNumberOfWrittenBits();
          Double splitBitCost = m_pcRdCost->calcRdCost( splitBits, 0 );
          splitTotalCost += splitBitCost;
        }

        rpcTempCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // split bits
        rpcTempCU->getTotalBins() += ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
      }

      if ( m_pcEncCfg->getLumaLevelToDeltaQPMapping().isEnabled() && pps.getMaxCuDQPDepth() >= 1 )
      {
        rpcTempCU->getTotalCost() = splitTotalCost;
      }
      else
      {
        rpcTempCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );
      }

      if( uiDepth == pps.getMaxCuDQPDepth() && pps.getUseDQP())
      {
        Bool hasResidual = false;
        for( UInt uiBlkIdx = 0; uiBlkIdx < rpcTempCU->getTotalNumPart(); uiBlkIdx ++)
        {
          if( (     rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Y)
                || (rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Cb) && (numberValidComponents > COMPONENT_Cb))
                || (rpcTempCU->getCbf(uiBlkIdx, COMPONENT_Cr) && (numberValidComponents > COMPONENT_Cr)) ) 
                || ( rpcTempCU->getPaletteModeFlag(uiBlkIdx) && rpcTempCU->getPaletteEscape(COMPONENT_Y, uiBlkIdx) ) )
          {
            hasResidual = true;
            break;
          }
        }

        if ( hasResidual )
        {
          m_pcEntropyCoder->resetBits();
          m_pcEntropyCoder->encodeQP( rpcTempCU, 0, false );
          rpcTempCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // dQP bits
          rpcTempCU->getTotalBins() += ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
          rpcTempCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

          Bool foundNonZeroCbf = false;
          rpcTempCU->setQPSubCUs( rpcTempCU->getRefQP( 0 ), 0, uiDepth, foundNonZeroCbf );
          assert( foundNonZeroCbf );
        }
        else
        {
          rpcTempCU->setQPSubParts( rpcTempCU->getRefQP( 0 ), 0, uiDepth ); // set QP to default QP
        }
      }

      m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]);

      // If the configuration being tested exceeds the maximum number of bytes for a slice / slice-segment, then
      // a proper RD evaluation cannot be performed. Therefore, termination of the
      // slice/slice-segment must be made prior to this CTU.
      // This can be achieved by forcing the decision to be that of the rpcTempCU.
      // The exception is each slice / slice-segment must have at least one CTU.
      if (rpcBestCU->getTotalCost()!=MAX_DOUBLE)
      {
        const Bool isEndOfSlice        =    pcSlice->getSliceMode()==FIXED_NUMBER_OF_BYTES
                                         && ((pcSlice->getSliceBits()+rpcBestCU->getTotalBits())>pcSlice->getSliceArgument()<<3)
                                         && rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceCurStartCtuTsAddr())
                                         && rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceSegmentCurStartCtuTsAddr());
        const Bool isEndOfSliceSegment =    pcSlice->getSliceSegmentMode()==FIXED_NUMBER_OF_BYTES
                                         && ((pcSlice->getSliceSegmentBits()+rpcBestCU->getTotalBits()) > pcSlice->getSliceSegmentArgument()<<3)
                                         && rpcBestCU->getCtuRsAddr() != pcPic->getPicSym()->getCtuTsToRsAddrMap(pcSlice->getSliceSegmentCurStartCtuTsAddr());
                                             // Do not need to check slice condition for slice-segment since a slice-segment is a subset of a slice.
        if(isEndOfSlice||isEndOfSliceSegment)
        {
          rpcBestCU->getTotalCost()=MAX_DOUBLE;
        }
      }

      xCheckBestMode( rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTempDebug) DEBUG_STRING_PASS_INTO(false) ); // RD compare current larger prediction
                                                                                                                                                       // with sub partitioned prediction.
    }
  }

  DEBUG_STRING_APPEND(sDebug_, sDebug);

  rpcBestCU->copyToPic(uiDepth);                                                     // Copy Best data to Picture for next partition prediction.

  xCopyYuv2Pic( rpcBestCU->getPic(), rpcBestCU->getCtuRsAddr(), rpcBestCU->getZorderIdxInCtu(), uiDepth, uiDepth, rpcBestCU );   // Copy Yuv data to picture Yuv
  for (UInt ch = 0; ch < numValidComp; ch++)
  {
    if (lastPalette[ch])
    {
      xFree(lastPalette[ch]);
      lastPalette[ch] = NULL;
    }
  }
  if (bBoundary)
  {
    return;
  }

  // Assert if Best prediction mode is NONE
  // Selected mode's RD-cost must be not MAX_DOUBLE.
  assert( rpcBestCU->getPartitionSize ( 0 ) != NUMBER_OF_PART_SIZES       );
  assert( rpcBestCU->getPredictionMode( 0 ) != NUMBER_OF_PREDICTION_MODES );
  assert( rpcBestCU->getTotalCost     (   ) != MAX_DOUBLE                 );
}

/** finish encoding a cu and handle end-of-slice conditions
 * \param pcCU
 * \param uiAbsPartIdx
 * \param uiDepth
 * \returns Void
 */
Void TEncCu::finishCU( TComDataCU* pcCU, UInt uiAbsPartIdx )
{
  TComPic* pcPic = pcCU->getPic();
  TComSlice * pcSlice = pcCU->getPic()->getSlice(pcCU->getPic()->getCurrSliceIdx());

  //Calculate end address
  const Int  currentCTUTsAddr = pcPic->getPicSym()->getCtuRsToTsAddrMap(pcCU->getCtuRsAddr());
  const Bool isLastSubCUOfCtu = pcCU->isLastSubCUOfCtu(uiAbsPartIdx);
  if ( isLastSubCUOfCtu )
  {
    // The 1-terminating bit is added to all streams, so don't add it here when it's 1.
    // i.e. when the slice segment CurEnd CTU address is the current CTU address+1.
    if (pcSlice->getSliceSegmentCurEndCtuTsAddr() != currentCTUTsAddr+1)
    {
      m_pcEntropyCoder->encodeTerminatingBit( 0 );
    }
  }
}

/** Compute QP for each CU
 * \param pcCU Target CU
 * \param uiDepth CU depth
 * \returns quantization parameter
 */
Int TEncCu::xComputeQP( TComDataCU* pcCU, UInt uiDepth )
{
  Int iBaseQp = pcCU->getSlice()->getSliceQp();
  Int iQpOffset = 0;
  if ( m_pcEncCfg->getUseAdaptiveQP() )
  {
    TEncPic* pcEPic = dynamic_cast<TEncPic*>( pcCU->getPic() );
    UInt uiAQDepth = min( uiDepth, pcEPic->getMaxAQDepth()-1 );
    TEncPicQPAdaptationLayer* pcAQLayer = pcEPic->getAQLayer( uiAQDepth );
    UInt uiAQUPosX = pcCU->getCUPelX() / pcAQLayer->getAQPartWidth();
    UInt uiAQUPosY = pcCU->getCUPelY() / pcAQLayer->getAQPartHeight();
    UInt uiAQUStride = pcAQLayer->getAQPartStride();
    TEncQPAdaptationUnit* acAQU = pcAQLayer->getQPAdaptationUnit();

    Double dMaxQScale = pow(2.0, m_pcEncCfg->getQPAdaptationRange()/6.0);
    Double dAvgAct = pcAQLayer->getAvgActivity();
    Double dCUAct = acAQU[uiAQUPosY * uiAQUStride + uiAQUPosX].getActivity();
    Double dNormAct = (dMaxQScale*dCUAct + dAvgAct) / (dCUAct + dMaxQScale*dAvgAct);
    Double dQpOffset = log(dNormAct) / log(2.0) * 6.0;
    iQpOffset = Int(floor( dQpOffset + 0.49999 ));
  }

  return Clip3(-pcCU->getSlice()->getSPS()->getQpBDOffset(CHANNEL_TYPE_LUMA), MAX_QP, iBaseQp+iQpOffset );
}

/** encode a CU block recursively
 * \param pcCU
 * \param uiAbsPartIdx
 * \param uiDepth
 * \returns Void
 */
Void TEncCu::xEncodeCU( TComDataCU* pcCU, UInt uiAbsPartIdx, UInt uiDepth )
{
        TComPic   *const pcPic   = pcCU->getPic();
        TComSlice *const pcSlice = pcCU->getSlice();
  const TComSPS   &sps =*(pcSlice->getSPS());
  const TComPPS   &pps =*(pcSlice->getPPS());

  const UInt maxCUWidth  = sps.getMaxCUWidth();
  const UInt maxCUHeight = sps.getMaxCUHeight();

        Bool bBoundary = false;
        UInt uiLPelX   = pcCU->getCUPelX() + g_auiRasterToPelX[ g_auiZscanToRaster[uiAbsPartIdx] ];
  const UInt uiRPelX   = uiLPelX + (maxCUWidth>>uiDepth)  - 1;
        UInt uiTPelY   = pcCU->getCUPelY() + g_auiRasterToPelY[ g_auiZscanToRaster[uiAbsPartIdx] ];
  const UInt uiBPelY   = uiTPelY + (maxCUHeight>>uiDepth) - 1;

  if( ( uiRPelX < sps.getPicWidthInLumaSamples() ) && ( uiBPelY < sps.getPicHeightInLumaSamples() ) )
  {
    m_pcEntropyCoder->encodeSplitFlag( pcCU, uiAbsPartIdx, uiDepth );
  }
  else
  {
    bBoundary = true;
  }

  if( ( ( uiDepth < pcCU->getDepth( uiAbsPartIdx ) ) && ( uiDepth < sps.getLog2DiffMaxMinCodingBlockSize() ) ) || bBoundary )
  {
    UInt uiQNumParts = ( pcPic->getNumPartitionsInCtu() >> (uiDepth<<1) )>>2;
    if( uiDepth == pps.getMaxCuDQPDepth() && pps.getUseDQP())
    {
      setdQPFlag(true);
    }

    if( uiDepth == pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth() && pcSlice->getUseChromaQpAdj())
    {
      setCodeChromaQpAdjFlag(true);
    }

    for ( UInt uiPartUnitIdx = 0; uiPartUnitIdx < 4; uiPartUnitIdx++, uiAbsPartIdx+=uiQNumParts )
    {
      uiLPelX   = pcCU->getCUPelX() + g_auiRasterToPelX[ g_auiZscanToRaster[uiAbsPartIdx] ];
      uiTPelY   = pcCU->getCUPelY() + g_auiRasterToPelY[ g_auiZscanToRaster[uiAbsPartIdx] ];
      if( ( uiLPelX < sps.getPicWidthInLumaSamples() ) && ( uiTPelY < sps.getPicHeightInLumaSamples() ) )
      {
        xEncodeCU( pcCU, uiAbsPartIdx, uiDepth+1 );
      }
    }
    return;
  }

  if( uiDepth <= pps.getMaxCuDQPDepth() && pps.getUseDQP())
  {
    setdQPFlag(true);
  }

  if( uiDepth <= pps.getPpsRangeExtension().getDiffCuChromaQpOffsetDepth() && pcSlice->getUseChromaQpAdj())
  {
    setCodeChromaQpAdjFlag(true);
  }

  if (pps.getTransquantBypassEnabledFlag())
  {
    m_pcEntropyCoder->encodeCUTransquantBypassFlag( pcCU, uiAbsPartIdx );
  }

  if( !pcSlice->isIntra() )
  {
    m_pcEntropyCoder->encodeSkipFlag( pcCU, uiAbsPartIdx );
  }

  if( pcCU->isSkipped( uiAbsPartIdx ) )
  {
    m_pcEntropyCoder->encodeMergeIndex( pcCU, uiAbsPartIdx );
    finishCU(pcCU,uiAbsPartIdx);
    return;
  }

  m_pcEntropyCoder->encodePredMode( pcCU, uiAbsPartIdx );

  Bool bCodeDQP = getdQPFlag();
  Bool codeChromaQpAdj = getCodeChromaQpAdjFlag();

  m_pcEntropyCoder->encodePaletteModeInfo( pcCU, uiAbsPartIdx, false, &bCodeDQP, &codeChromaQpAdj );

  if ( pcCU->getPaletteModeFlag(uiAbsPartIdx) )
  {
    setCodeChromaQpAdjFlag( codeChromaQpAdj );
    setdQPFlag( bCodeDQP );
    finishCU(pcCU,uiAbsPartIdx);
    return;
  }

  m_pcEntropyCoder->encodePartSize( pcCU, uiAbsPartIdx, uiDepth );

  if (pcCU->isIntra( uiAbsPartIdx ) && pcCU->getPartitionSize( uiAbsPartIdx ) == SIZE_2Nx2N )
  {
    m_pcEntropyCoder->encodeIPCMInfo( pcCU, uiAbsPartIdx );

    if(pcCU->getIPCMFlag(uiAbsPartIdx))
    {
      // Encode slice finish
      finishCU(pcCU,uiAbsPartIdx);
      return;
    }
  }

  // prediction Info ( Intra : direction mode, Inter : Mv, reference idx )
  m_pcEntropyCoder->encodePredInfo( pcCU, uiAbsPartIdx );

  // Encode Coefficients
  m_pcEntropyCoder->encodeCoeff( pcCU, uiAbsPartIdx, uiDepth, bCodeDQP, codeChromaQpAdj );
  setCodeChromaQpAdjFlag( codeChromaQpAdj );
  setdQPFlag( bCodeDQP );

  // --- write terminating bit ---
  finishCU(pcCU,uiAbsPartIdx);
}

Int xCalcHADs8x8_ISlice(Pel *piOrg, Int iStrideOrg)
{
  Int k, i, j, jj;
  Int diff[64], m1[8][8], m2[8][8], m3[8][8], iSumHad = 0;

  for( k = 0; k < 64; k += 8 )
  {
    diff[k+0] = piOrg[0] ;
    diff[k+1] = piOrg[1] ;
    diff[k+2] = piOrg[2] ;
    diff[k+3] = piOrg[3] ;
    diff[k+4] = piOrg[4] ;
    diff[k+5] = piOrg[5] ;
    diff[k+6] = piOrg[6] ;
    diff[k+7] = piOrg[7] ;

    piOrg += iStrideOrg;
  }

  //horizontal
  for (j=0; j < 8; j++)
  {
    jj = j << 3;
    m2[j][0] = diff[jj  ] + diff[jj+4];
    m2[j][1] = diff[jj+1] + diff[jj+5];
    m2[j][2] = diff[jj+2] + diff[jj+6];
    m2[j][3] = diff[jj+3] + diff[jj+7];
    m2[j][4] = diff[jj  ] - diff[jj+4];
    m2[j][5] = diff[jj+1] - diff[jj+5];
    m2[j][6] = diff[jj+2] - diff[jj+6];
    m2[j][7] = diff[jj+3] - diff[jj+7];

    m1[j][0] = m2[j][0] + m2[j][2];
    m1[j][1] = m2[j][1] + m2[j][3];
    m1[j][2] = m2[j][0] - m2[j][2];
    m1[j][3] = m2[j][1] - m2[j][3];
    m1[j][4] = m2[j][4] + m2[j][6];
    m1[j][5] = m2[j][5] + m2[j][7];
    m1[j][6] = m2[j][4] - m2[j][6];
    m1[j][7] = m2[j][5] - m2[j][7];

    m2[j][0] = m1[j][0] + m1[j][1];
    m2[j][1] = m1[j][0] - m1[j][1];
    m2[j][2] = m1[j][2] + m1[j][3];
    m2[j][3] = m1[j][2] - m1[j][3];
    m2[j][4] = m1[j][4] + m1[j][5];
    m2[j][5] = m1[j][4] - m1[j][5];
    m2[j][6] = m1[j][6] + m1[j][7];
    m2[j][7] = m1[j][6] - m1[j][7];
  }

  //vertical
  for (i=0; i < 8; i++)
  {
    m3[0][i] = m2[0][i] + m2[4][i];
    m3[1][i] = m2[1][i] + m2[5][i];
    m3[2][i] = m2[2][i] + m2[6][i];
    m3[3][i] = m2[3][i] + m2[7][i];
    m3[4][i] = m2[0][i] - m2[4][i];
    m3[5][i] = m2[1][i] - m2[5][i];
    m3[6][i] = m2[2][i] - m2[6][i];
    m3[7][i] = m2[3][i] - m2[7][i];

    m1[0][i] = m3[0][i] + m3[2][i];
    m1[1][i] = m3[1][i] + m3[3][i];
    m1[2][i] = m3[0][i] - m3[2][i];
    m1[3][i] = m3[1][i] - m3[3][i];
    m1[4][i] = m3[4][i] + m3[6][i];
    m1[5][i] = m3[5][i] + m3[7][i];
    m1[6][i] = m3[4][i] - m3[6][i];
    m1[7][i] = m3[5][i] - m3[7][i];

    m2[0][i] = m1[0][i] + m1[1][i];
    m2[1][i] = m1[0][i] - m1[1][i];
    m2[2][i] = m1[2][i] + m1[3][i];
    m2[3][i] = m1[2][i] - m1[3][i];
    m2[4][i] = m1[4][i] + m1[5][i];
    m2[5][i] = m1[4][i] - m1[5][i];
    m2[6][i] = m1[6][i] + m1[7][i];
    m2[7][i] = m1[6][i] - m1[7][i];
  }

  for (i = 0; i < 8; i++)
  {
    for (j = 0; j < 8; j++)
    {
      iSumHad += abs(m2[i][j]);
    }
  }
  iSumHad -= abs(m2[0][0]);
  iSumHad =(iSumHad+2)>>2;
  return(iSumHad);
}

Int  TEncCu::updateCtuDataISlice(TComDataCU* pCtu, Int width, Int height)
{
  Int  xBl, yBl;
  const Int iBlkSize = 8;

  Pel* pOrgInit   = pCtu->getPic()->getPicYuvOrg()->getAddr(COMPONENT_Y, pCtu->getCtuRsAddr(), 0);
  Int  iStrideOrig = pCtu->getPic()->getPicYuvOrg()->getStride(COMPONENT_Y);
  Pel  *pOrg;

  Int iSumHad = 0;
  for ( yBl=0; (yBl+iBlkSize)<=height; yBl+= iBlkSize)
  {
    for ( xBl=0; (xBl+iBlkSize)<=width; xBl+= iBlkSize)
    {
      pOrg = pOrgInit + iStrideOrig*yBl + xBl;
      iSumHad += xCalcHADs8x8_ISlice(pOrg, iStrideOrig);
    }
  }
  return(iSumHad);
}

/** check RD costs for a CU block encoded with merge
 * \param rpcBestCU
 * \param rpcTempCU
 * \param earlyDetectionSkipMode
 */
Void TEncCu::xCheckRDCostMerge2Nx2N( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU DEBUG_STRING_FN_DECLARE(sDebug), Bool *earlyDetectionSkipMode, Bool checkSkipOnly )
{
  assert( rpcTempCU->getSlice()->getSliceType() != I_SLICE );
  if(getFastDeltaQp())
  {
    return;   // never check merge in fast deltaqp mode
  }
  TComMvField  cMvFieldNeighbours[2 * MRG_MAX_NUM_CANDS]; // double length for mv of both lists
  UChar uhInterDirNeighbours[MRG_MAX_NUM_CANDS];
  Int numValidMergeCand = 0;
  const Bool bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);

  for( UInt ui = 0; ui < rpcTempCU->getSlice()->getMaxNumMergeCand(); ++ui )
  {
    uhInterDirNeighbours[ui] = 0;
  }
  UChar uhDepth = rpcTempCU->getDepth( 0 );
  rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, uhDepth ); // interprets depth relative to CTU level
#if MCTS_ENC_CHECK
  UInt numSpatialMergeCandidates = 0;
  rpcTempCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
#else
  rpcTempCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours,uhInterDirNeighbours, numValidMergeCand );
#endif
  rpcTempCU->roundMergeCandidates(cMvFieldNeighbours, numValidMergeCand);
  rpcTempCU->xRestrictBipredMergeCand(0, cMvFieldNeighbours, uhInterDirNeighbours, numValidMergeCand );

  Int mergeCandBuffer[MRG_MAX_NUM_CANDS];
  for( UInt ui = 0; ui < numValidMergeCand; ++ui )
  {
    mergeCandBuffer[ui] = 0;
  }
#if MCTS_ENC_CHECK
  if (m_pcEncCfg->getTMCTSSEITileConstraint() && rpcTempCU->isLastColumnCTUInTile())
  {
    numValidMergeCand = numSpatialMergeCandidates;
  }
#endif

  Bool bestIsSkip = false;

  UInt iteration;
  UInt iterationBegin = checkSkipOnly ? 1 : 0;
  if ( rpcTempCU->isLosslessCoded(0))
  {
    iteration = 1;
    iterationBegin = 0;
  }
  else
  {
    iteration = 2;
  }
  DEBUG_STRING_NEW(bestStr)

  const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
  for( UInt uiNoResidual = iterationBegin; uiNoResidual < iteration; ++uiNoResidual )
  {
    for( UInt uiMergeCand = 0; uiMergeCand < numValidMergeCand; ++uiMergeCand )
    {
      if(!(uiNoResidual==1 && mergeCandBuffer[uiMergeCand]==1))
      {
        if( !(bestIsSkip && uiNoResidual == 0) )
        {
          if ( (uhInterDirNeighbours[uiMergeCand] == 1 || uhInterDirNeighbours[uiMergeCand] == 3) && rpcTempCU->getSlice()->getRefPic( REF_PIC_LIST_0, cMvFieldNeighbours[uiMergeCand<<1].getRefIdx() )->getPOC() == rpcTempCU->getSlice()->getPOC() )
          {
            continue;
          }

          DEBUG_STRING_NEW(tmpStr)
          // set MC parameters
          rpcTempCU->setPredModeSubParts( MODE_INTER, 0, uhDepth ); // interprets depth relative to CTU level
          rpcTempCU->setCUTransquantBypassSubParts( bTransquantBypassFlag, 0, uhDepth );
          rpcTempCU->setChromaQpAdjSubParts( bTransquantBypassFlag ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uhDepth );
          rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, uhDepth ); // interprets depth relative to CTU level
          rpcTempCU->setMergeFlagSubParts( true, 0, 0, uhDepth ); // interprets depth relative to CTU level
          rpcTempCU->setMergeIndexSubParts( uiMergeCand, 0, 0, uhDepth ); // interprets depth relative to CTU level
          rpcTempCU->setInterDirSubParts( uhInterDirNeighbours[uiMergeCand], 0, 0, uhDepth ); // interprets depth relative to CTU level
          rpcTempCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMvFieldNeighbours[0 + 2*uiMergeCand], SIZE_2Nx2N, 0, 0 ); // interprets depth relative to rpcTempCU level
          rpcTempCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMvFieldNeighbours[1 + 2*uiMergeCand], SIZE_2Nx2N, 0, 0 ); // interprets depth relative to rpcTempCU level

#if MCTS_ENC_CHECK
          if ( m_pcEncCfg->getTMCTSSEITileConstraint () && (!(m_pcPredSearch->checkTMctsMvp(rpcTempCU))))
          {
            continue;
          }

#endif
          // do MC
          m_pcPredSearch->motionCompensation ( rpcTempCU, m_ppcPredYuvTemp[uhDepth] );
          Bool bColourTrans = (m_pcEncCfg->getRGBFormatFlag() && rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans())? true : false;

          if ( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
          {
            bColourTrans = false;
          }
          TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[uhDepth];
          if( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && (!bTransquantBypassFlag || sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
          {
            if(!getEnableInterTUACT())
            {
              m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                         m_ppcOrigYuv    [uhDepth],
                                                         pcTmpPredYuv             ,
                                                         m_ppcResiYuvTemp[uhDepth],
                                                         m_ppcResiYuvBest[uhDepth],
                                                         m_ppcRecoYuvTemp[uhDepth],
                                                         (uiNoResidual != 0),
                                                         m_ppcNoCorrYuv  [uhDepth],
                                                         (bColourTrans? ACT_TRAN_CLR: ACT_ORG_CLR)
                                                         DEBUG_STRING_PASS_INTO(tmpStr) );
            }
            else
            {
              m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                         m_ppcOrigYuv    [uhDepth],
                                                         pcTmpPredYuv             ,
                                                         m_ppcResiYuvTemp[uhDepth],
                                                         m_ppcResiYuvBest[uhDepth],
                                                         m_ppcRecoYuvTemp[uhDepth],
                                                         (uiNoResidual != 0),
                                                         m_ppcNoCorrYuv  [uhDepth],
                                                         ACT_TWO_CLR
                                                         DEBUG_STRING_PASS_INTO(tmpStr) );
            }
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                       m_ppcOrigYuv    [uhDepth],
                                                       pcTmpPredYuv             ,
                                                       m_ppcResiYuvTemp[uhDepth],
                                                       m_ppcResiYuvBest[uhDepth],
                                                       m_ppcRecoYuvTemp[uhDepth],
                                                       (uiNoResidual != 0),
                                                       m_ppcNoCorrYuv  [uhDepth],
                                                       ACT_ORG_CLR
                                                       DEBUG_STRING_PASS_INTO(tmpStr) );
          }
          rpcTempCU->setSkipFlagSubParts( rpcTempCU->getQtRootCbf(0) == 0, 0, uhDepth );

#if DEBUG_STRING
          DebugInterPredResiReco(tmpStr, *(m_ppcPredYuvTemp[uhDepth]), *(m_ppcResiYuvBest[uhDepth]), *(m_ppcRecoYuvTemp[uhDepth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif

          if ((uiNoResidual == 0) && (rpcTempCU->getQtRootCbf(0) == 0))
          {
            // If no residual when allowing for one, then set mark to not try case where residual is forced to 0
            mergeCandBuffer[uiMergeCand] = 1;
          }

          Double rdCost1 = rpcTempCU->getTotalCost();
          if ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() )
          {
            if ( rdCost1 < m_ppcBestCU[uhDepth]->getTmpInterRDCost() )
            {
              m_ppcBestCU[uhDepth]->setTmpInterRDCost(rdCost1);
              m_ppcBestCU[uhDepth]->setInterCSCEnabled(true);
              m_ppcTempCU[uhDepth]->setTmpInterRDCost(rdCost1);
              m_ppcTempCU[uhDepth]->setInterCSCEnabled(true);
            }
          }

          Int orgQP = rpcTempCU->getQP( 0 );
          xCheckDQP( rpcTempCU );
          TComDataCU *rpcTempCUPre = rpcTempCU;

          xCheckBestMode(rpcBestCU, rpcTempCU, uhDepth DEBUG_STRING_PASS_INTO(bestStr) DEBUG_STRING_PASS_INTO(tmpStr));

          Bool bParentUseCSC = ( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans()
                                 && (uhDepth != 0) && (m_ppcBestCU[uhDepth -1]->getInterCSCEnabled()) );
          bParentUseCSC = bParentUseCSC || rpcBestCU->getQtRootCbf(0) == 0 || rpcBestCU->getMergeFlag( 0 );
          if(rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && !uiNoResidual && rpcTempCUPre->getQtRootCbf(0) && !bParentUseCSC && (!bTransquantBypassFlag || sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
          {
            if( rpcTempCU != rpcTempCUPre )
            {
              rpcTempCU->initEstData( uhDepth, orgQP, bTransquantBypassFlag );
              rpcTempCU->copyPartFrom( rpcBestCU, 0, uhDepth );
            }
            if ( !getEnableInterTUACT() )
            {
              bColourTrans = !bColourTrans;
            }
            else
            {
              bColourTrans = m_pcEncCfg->getRGBFormatFlag();
            }
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                       m_ppcOrigYuv    [uhDepth],
                                                       pcTmpPredYuv             ,
                                                       m_ppcResiYuvTemp[uhDepth],
                                                       m_ppcResiYuvBest[uhDepth],
                                                       m_ppcRecoYuvTemp[uhDepth],
                                                       (uiNoResidual != 0),
                                                       m_ppcNoCorrYuv  [uhDepth],
                                                       (bColourTrans? ACT_TRAN_CLR: ACT_ORG_CLR)
                                                       DEBUG_STRING_PASS_INTO(tmpStr) );
            rpcTempCU->setSkipFlagSubParts( rpcTempCU->getQtRootCbf(0) == 0, 0, uhDepth );
            Double rdCost = rpcTempCU->getTotalCost();
            if(rdCost < m_ppcBestCU[uhDepth]->getTmpInterRDCost() )
            {
              m_ppcBestCU[uhDepth]->setTmpInterRDCost(rdCost);
              m_ppcBestCU[uhDepth]->setInterCSCEnabled(false);
              m_ppcTempCU[uhDepth]->setTmpInterRDCost(rdCost);
              m_ppcTempCU[uhDepth]->setInterCSCEnabled(false);
            }

            xCheckDQP( rpcTempCU );
            if(rpcTempCU->getQtRootCbf(0))
            {
              xCheckBestMode( rpcBestCU, rpcTempCU, uhDepth DEBUG_STRING_PASS_INTO(bestStr) DEBUG_STRING_PASS_INTO(tmpStr) );
            }
          }
          rpcTempCU->initEstData( uhDepth, orgQP, bTransquantBypassFlag );

          if( m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip )
          {
            bestIsSkip = rpcBestCU->getQtRootCbf(0) == 0;
          }
        }
      }
    }

    if(uiNoResidual == 0 && m_pcEncCfg->getUseEarlySkipDetection())
    {
      if(rpcBestCU->getQtRootCbf( 0 ) == 0)
      {
        if( rpcBestCU->getMergeFlag( 0 ))
        {
          *earlyDetectionSkipMode = true;
        }
        else if(m_pcEncCfg->getMotionEstimationSearchMethod() != MESEARCH_SELECTIVE)
        {
          Int absoulte_MV=0;
          for ( UInt uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
          {
            if ( rpcBestCU->getSlice()->getNumRefIdx( RefPicList( uiRefListIdx ) ) > 0 )
            {
              TComCUMvField* pcCUMvField = rpcBestCU->getCUMvField(RefPicList( uiRefListIdx ));
              Int iHor = pcCUMvField->getMvd( 0 ).getAbsHor();
              Int iVer = pcCUMvField->getMvd( 0 ).getAbsVer();
              absoulte_MV+=iHor+iVer;
            }
          }

          if(absoulte_MV == 0)
          {
            *earlyDetectionSkipMode = true;
          }
        }
      }
    }
  }
  DEBUG_STRING_APPEND(sDebug, bestStr)
}


#if AMP_MRG
Void TEncCu::xCheckRDCostInter( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, PartSize ePartSize DEBUG_STRING_FN_DECLARE(sDebug), Bool bUseMRG, TComMv *iMVCandList)
#else
Void TEncCu::xCheckRDCostInter( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, PartSize ePartSize, TComMv *iMVCandList )
#endif
{
  DEBUG_STRING_NEW(sTest)

  if(getFastDeltaQp())
  {
    const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
    const UInt fastDeltaQPCuMaxSize = Clip3(sps.getMaxCUHeight()>>(sps.getLog2DiffMaxMinCodingBlockSize()), sps.getMaxCUHeight(), 32u);
    if(ePartSize != SIZE_2Nx2N || rpcTempCU->getWidth( 0 ) > fastDeltaQPCuMaxSize)
    {
      return; // only check necessary 2Nx2N Inter in fast deltaqp mode
    }
  }

  // prior to this, rpcTempCU will have just been reset using rpcTempCU->initEstData( uiDepth, iQP, bIsLosslessMode );
  UChar uhDepth = rpcTempCU->getDepth( 0 );

  rpcTempCU->setPartSizeSubParts  ( ePartSize,  0, uhDepth );
  rpcTempCU->setPredModeSubParts  ( MODE_INTER, 0, uhDepth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uhDepth );

#if MCTS_ENC_CHECK
  rpcTempCU->setTMctsMvpIsValid(true);
#endif

#if AMP_MRG
  rpcTempCU->setMergeAMP (true);
  Bool valid = m_pcPredSearch->predInterSearch ( rpcTempCU, m_ppcOrigYuv[uhDepth], m_ppcPredYuvTemp[uhDepth], m_ppcResiYuvTemp[uhDepth], m_ppcRecoYuvTemp[uhDepth] DEBUG_STRING_PASS_INTO(sTest), false, bUseMRG, iMVCandList );
#else
  Bool valid = m_pcPredSearch->predInterSearch ( rpcTempCU, m_ppcOrigYuv[uhDepth], m_ppcPredYuvTemp[uhDepth], m_ppcResiYuvTemp[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, iMVCandList );
#endif

  if( !valid )
  {
    return;
  }

#if AMP_MRG
  if ( !rpcTempCU->getMergeAMP() )
  {
    return;
  }
#endif

#if MCTS_ENC_CHECK
  if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!rpcTempCU->getTMctsMvpIsValid()))
  {
    return;
  }
#endif

  TComDataCU *rpcTempCUPre = NULL;
  Bool   bEnableTrans      = rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans();

  UChar  colourTransform = 0;
  Bool   bRGB              = m_pcEncCfg->getRGBFormatFlag();
  SChar  orgQP             = rpcTempCU->getQP( 0 );
  Bool   bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);
  const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
  if ( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
  {
    bEnableTrans = false;
  }
  TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[uhDepth];
  for(UInt i = 0;  i < 2 ; i++)
  {
    colourTransform = (bRGB && bEnableTrans)? (1-i): i;
    if ( i == 0 )
    {
      if ( bEnableTrans )
      {
        if ( !getEnableInterTUACT() )
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], (colourTransform? ACT_TRAN_CLR: ACT_ORG_CLR) DEBUG_STRING_PASS_INTO(sTest) );
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_TWO_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
      else
      {
        m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
      }
    }
    else
    {
      if ( m_pcEncCfg->getRGBFormatFlag() )
      {
        if ( !getEnableInterTUACT() )
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
      else
      {
        if ( !getEnableInterTUACT() )
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[uhDepth], pcTmpPredYuv, m_ppcResiYuvTemp[uhDepth], m_ppcResiYuvBest[uhDepth], m_ppcRecoYuvTemp[uhDepth], false, m_ppcNoCorrYuv[uhDepth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
    }
    rpcTempCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

#if DEBUG_STRING
    DebugInterPredResiReco(sTest, *(m_ppcPredYuvTemp[uhDepth]), *(m_ppcResiYuvBest[uhDepth]), *(m_ppcRecoYuvTemp[uhDepth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif

    xCheckDQP( rpcTempCU );

    if ( bEnableTrans )
    {
      if ( !i )
      {
        Double rdCost1 = rpcTempCU->getTotalCost();
        if ( rdCost1 < m_ppcBestCU[uhDepth]->getTmpInterRDCost() )
        {
          m_ppcBestCU[uhDepth]->setTmpInterRDCost(rdCost1);
          m_ppcBestCU[uhDepth]->setInterCSCEnabled(true);
          m_ppcTempCU[uhDepth]->setTmpInterRDCost(rdCost1);
          m_ppcTempCU[uhDepth]->setInterCSCEnabled(true);
        }
      }
      else
      {
        Double rdCost = rpcTempCU->getTotalCost();
        if ( rdCost < m_ppcBestCU[uhDepth]->getTmpInterRDCost() )
        {
          m_ppcBestCU[uhDepth]->setTmpInterRDCost(rdCost);
          m_ppcBestCU[uhDepth]->setInterCSCEnabled(false);
          m_ppcTempCU[uhDepth]->setTmpInterRDCost(rdCost);
          m_ppcTempCU[uhDepth]->setInterCSCEnabled(false);
        }
      }
    }
    rpcTempCUPre = rpcTempCU;

    xCheckBestMode(rpcBestCU, rpcTempCU, uhDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));

    if(bEnableTrans)
    {
      if( !rpcBestCU->isInter(0) || rpcBestCU->isSkipped(0) )
      {
        return;
      }
      Bool bParentUseCSC = ( (uhDepth != 0) && (m_ppcBestCU[uhDepth -1]->getInterCSCEnabled()) );
      if(!i && (!rpcTempCUPre->getQtRootCbf(0) || bParentUseCSC))
      {
        return;
      }
      else if(!i && rpcTempCUPre->getQtRootCbf(0))
      {
        if( rpcTempCU != rpcTempCUPre )
        {
          rpcTempCU->initEstData( uhDepth, orgQP, bTransquantBypassFlag);
          rpcTempCU->copyPartFrom( rpcBestCU, 0, uhDepth );
        }
      }
    }
    else
    {
      return;
    }
  }
}

Void TEncCu::xCheckRDCostIntra( TComDataCU *&rpcBestCU,
                                TComDataCU *&rpcTempCU,
                                Double      &cost,
                                PartSize     eSize
                                DEBUG_STRING_FN_DECLARE(sDebug),
                                Bool         bRGBIntraModeReuse
                               )
{
  DEBUG_STRING_NEW(sTest)

  if(getFastDeltaQp())
  {
    const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
    const UInt fastDeltaQPCuMaxSize = Clip3(sps.getMaxCUHeight()>>(sps.getLog2DiffMaxMinCodingBlockSize()), sps.getMaxCUHeight(), 32u);
    if(rpcTempCU->getWidth( 0 ) > fastDeltaQPCuMaxSize)
    {
      return; // only check necessary 2Nx2N Intra in fast deltaqp mode
    }
  }

  UInt uiDepth = rpcTempCU->getDepth( 0 );

  rpcTempCU->setSkipFlagSubParts( false, 0, uiDepth );

  rpcTempCU->setPartSizeSubParts( eSize, 0, uiDepth );
  rpcTempCU->setPredModeSubParts( MODE_INTRA, 0, uiDepth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uiDepth );
  rpcTempCU->setColourTransformSubParts(false, 0 ,uiDepth);

  Pel resiLuma[NUMBER_OF_STORED_RESIDUAL_TYPES][MAX_CU_SIZE * MAX_CU_SIZE];

  if( bRGBIntraModeReuse )
  {
    m_pcPredSearch->estIntraPredLumaQTWithModeReuse( rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma );
  }
  else
  {
    m_pcPredSearch->estIntraPredLumaQT( rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma DEBUG_STRING_PASS_INTO(sTest) );
  }

  m_ppcRecoYuvTemp[uiDepth]->copyToPicComponent(COMPONENT_Y, rpcTempCU->getPic()->getPicYuvRec(), rpcTempCU->getCtuRsAddr(), rpcTempCU->getZorderIdxInCtu() );

  if (rpcBestCU->getPic()->getChromaFormat()!=CHROMA_400)
  {
    if( bRGBIntraModeReuse )
    {
      m_pcPredSearch->estIntraPredChromaQTWithModeReuse( rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma );
    }
    else
    {
      m_pcPredSearch->estIntraPredChromaQT( rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth], resiLuma DEBUG_STRING_PASS_INTO(sTest) );
    }
  }

  m_pcEntropyCoder->resetBits();

  if ( rpcTempCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
  {
    m_pcEntropyCoder->encodeCUTransquantBypassFlag( rpcTempCU, 0,          true );
  }

  m_pcEntropyCoder->encodeSkipFlag ( rpcTempCU, 0,          true );
  m_pcEntropyCoder->encodePredMode( rpcTempCU, 0,          true );
  Bool bCodeDQP = getdQPFlag();
  Bool codeChromaQpAdjFlag = getCodeChromaQpAdjFlag();

  m_pcEntropyCoder->encodePaletteModeInfo( rpcTempCU, 0, true, &bCodeDQP, &codeChromaQpAdjFlag );

  m_pcEntropyCoder->encodePartSize( rpcTempCU, 0, uiDepth, true );
  m_pcEntropyCoder->encodePredInfo( rpcTempCU, 0 );
  m_pcEntropyCoder->encodeIPCMInfo(rpcTempCU, 0, true );

  // Encode Coefficients
  m_pcEntropyCoder->encodeCoeff( rpcTempCU, 0, uiDepth, bCodeDQP, codeChromaQpAdjFlag );
  setCodeChromaQpAdjFlag( codeChromaQpAdjFlag );
  setdQPFlag( bCodeDQP );

  m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]);

  rpcTempCU->getTotalBits() = m_pcEntropyCoder->getNumberOfWrittenBits();
  rpcTempCU->getTotalBins() = ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
  rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

  xCheckDQP( rpcTempCU );

  cost = rpcTempCU->getTotalCost();

  xCheckBestMode(rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
}


/** Check R-D costs for a CU with PCM mode.
 * \param rpcBestCU pointer to best mode CU data structure
 * \param rpcTempCU pointer to testing mode CU data structure
 * \returns Void
 *
 * \note Current PCM implementation encodes sample values in a lossless way. The distortion of PCM mode CUs are zero. PCM mode is selected if the best mode yields bits greater than that of PCM mode.
 */
Void TEncCu::xCheckIntraPCM( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU )
{
  if(getFastDeltaQp())
  {
    const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
    const UInt fastDeltaQPCuMaxPCMSize = Clip3((UInt)1<<sps.getPCMLog2MinSize(), (UInt)1<<sps.getPCMLog2MaxSize(), 32u);
    if (rpcTempCU->getWidth( 0 ) > fastDeltaQPCuMaxPCMSize)
    {
      return;   // only check necessary PCM in fast deltaqp mode
    }
  }
  
  UInt uiDepth = rpcTempCU->getDepth( 0 );

  rpcTempCU->setSkipFlagSubParts( false, 0, uiDepth );

  rpcTempCU->setIPCMFlag(0, true);
  rpcTempCU->setIPCMFlagSubParts (true, 0, rpcTempCU->getDepth(0));
  rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, uiDepth );
  rpcTempCU->setPredModeSubParts( MODE_INTRA, 0, uiDepth );
  rpcTempCU->setTrIdxSubParts ( 0, 0, uiDepth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, uiDepth );

  m_pcPredSearch->IPCMSearch( rpcTempCU, m_ppcOrigYuv[uiDepth], m_ppcPredYuvTemp[uiDepth], m_ppcResiYuvTemp[uiDepth], m_ppcRecoYuvTemp[uiDepth]);

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[uiDepth][CI_CURR_BEST]);

  m_pcEntropyCoder->resetBits();

  if ( rpcTempCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag())
  {
    m_pcEntropyCoder->encodeCUTransquantBypassFlag( rpcTempCU, 0,          true );
  }

  m_pcEntropyCoder->encodeSkipFlag ( rpcTempCU, 0,          true );
  m_pcEntropyCoder->encodePredMode ( rpcTempCU, 0,          true );
  m_pcEntropyCoder->encodePaletteModeInfo( rpcTempCU, 0,    true );
  m_pcEntropyCoder->encodePartSize ( rpcTempCU, 0, uiDepth, true );
  m_pcEntropyCoder->encodeIPCMInfo ( rpcTempCU, 0, true );

  m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]);

  rpcTempCU->getTotalBits() = m_pcEntropyCoder->getNumberOfWrittenBits();
  rpcTempCU->getTotalBins() = ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
  rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

  xCheckDQP( rpcTempCU );
  DEBUG_STRING_NEW(a)
  DEBUG_STRING_NEW(b)
  xCheckBestMode(rpcBestCU, rpcTempCU, uiDepth DEBUG_STRING_PASS_INTO(a) DEBUG_STRING_PASS_INTO(b));
}

/** check whether current try is the best with identifying the depth of current try
 * \param rpcBestCU
 * \param rpcTempCU
 * \param uiDepth
 */
Void TEncCu::xCheckBestMode( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, UInt uiDepth DEBUG_STRING_FN_DECLARE(sParent) DEBUG_STRING_FN_DECLARE(sTest) DEBUG_STRING_PASS_INTO(Bool bAddSizeInfo) )
{
  if( rpcTempCU->getTotalCost() < rpcBestCU->getTotalCost() )
  {
    TComYuv* pcYuv;
    // Change Information data
    TComDataCU* pcCU = rpcBestCU;
    rpcBestCU = rpcTempCU;
    rpcTempCU = pcCU;

    // Change Prediction data
    pcYuv = m_ppcPredYuvBest[uiDepth];
    m_ppcPredYuvBest[uiDepth] = m_ppcPredYuvTemp[uiDepth];
    m_ppcPredYuvTemp[uiDepth] = pcYuv;

    // Change Reconstruction data
    pcYuv = m_ppcRecoYuvBest[uiDepth];
    m_ppcRecoYuvBest[uiDepth] = m_ppcRecoYuvTemp[uiDepth];
    m_ppcRecoYuvTemp[uiDepth] = pcYuv;

    pcYuv = NULL;
    pcCU  = NULL;

    // store temp best CI for next CU coding
    m_pppcRDSbacCoder[uiDepth][CI_TEMP_BEST]->store(m_pppcRDSbacCoder[uiDepth][CI_NEXT_BEST]);


#if DEBUG_STRING
    DEBUG_STRING_SWAP(sParent, sTest)
    const PredMode predMode=rpcBestCU->getPredictionMode(0);
    if ((DebugOptionList::DebugString_Structure.getInt()&DebugStringGetPredModeMask(predMode)) && bAddSizeInfo)
    {
      std::stringstream ss(stringstream::out);
      ss <<"###: " << (predMode==MODE_INTRA?"Intra   ":"Inter   ") << partSizeToString[rpcBestCU->getPartitionSize(0)] << " CU at " << rpcBestCU->getCUPelX() << ", " << rpcBestCU->getCUPelY() << " width=" << UInt(rpcBestCU->getWidth(0)) << std::endl;
      sParent+=ss.str();
    }
#endif
  }
}

Void TEncCu::xCheckDQP( TComDataCU* pcCU )
{
  UInt uiDepth = pcCU->getDepth( 0 );

  const TComPPS &pps = *(pcCU->getSlice()->getPPS());
  if ( pps.getUseDQP() && uiDepth <= pps.getMaxCuDQPDepth() )
  {
    if ( pcCU->getQtRootCbf( 0) || ( pcCU->getPaletteModeFlag(0) && pcCU->getPaletteEscape(COMPONENT_Y, 0) ) )
    {
      m_pcEntropyCoder->resetBits();
      m_pcEntropyCoder->encodeQP( pcCU, 0, false );
      pcCU->getTotalBits() += m_pcEntropyCoder->getNumberOfWrittenBits(); // dQP bits
      pcCU->getTotalBins() += ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
      pcCU->getTotalCost() = m_pcRdCost->calcRdCost( pcCU->getTotalBits(), pcCU->getTotalDistortion() );
    }
    else
    {
      pcCU->setQPSubParts( pcCU->getRefQP( 0 ), 0, uiDepth ); // set QP to default QP
    }
  }
}

Void TEncCu::xCopyAMVPInfo (AMVPInfo* pSrc, AMVPInfo* pDst)
{
  pDst->iN = pSrc->iN;
  for (Int i = 0; i < pSrc->iN; i++)
  {
    pDst->m_acMvCand[i] = pSrc->m_acMvCand[i];
  }
}
Void TEncCu::xCopyYuv2Pic(TComPic* rpcPic, UInt uiCUAddr, UInt uiAbsPartIdx, UInt uiDepth, UInt uiSrcDepth, TComDataCU* pcCU )
{
  UInt uiAbsPartIdxInRaster = g_auiZscanToRaster[uiAbsPartIdx];
  UInt uiSrcBlkWidth = rpcPic->getNumPartInCtuWidth() >> (uiSrcDepth);
  UInt uiBlkWidth    = rpcPic->getNumPartInCtuWidth() >> (uiDepth);
  UInt uiPartIdxX = ( ( uiAbsPartIdxInRaster % rpcPic->getNumPartInCtuWidth() ) % uiSrcBlkWidth) / uiBlkWidth;
  UInt uiPartIdxY = ( ( uiAbsPartIdxInRaster / rpcPic->getNumPartInCtuWidth() ) % uiSrcBlkWidth) / uiBlkWidth;
  UInt uiPartIdx = uiPartIdxY * ( uiSrcBlkWidth / uiBlkWidth ) + uiPartIdxX;
  m_ppcRecoYuvBest[uiSrcDepth]->copyToPicYuv( rpcPic->getPicYuvRec (), uiCUAddr, uiAbsPartIdx, uiDepth - uiSrcDepth, uiPartIdx);
  if( pcCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && m_pcEncCfg->getRGBFormatFlag() )
  {
    TComRectangle cuCompRect;
    cuCompRect.x0     = 0;
    cuCompRect.y0     = 0;
    cuCompRect.width  = (pcCU->getSlice()->getSPS()->getMaxCUWidth()>>uiSrcDepth);
    cuCompRect.height = (pcCU->getSlice()->getSPS()->getMaxCUHeight()>>uiSrcDepth);

    for ( UInt ch = 0; ch < MAX_NUM_COMPONENT; ch++ )
    {
      m_ppcRecoYuvBest[uiSrcDepth]->copyPartToPartComponentMxN( ComponentID( ch ), m_ppcRecoYuvTemp[uiSrcDepth], cuCompRect );
    }
    m_ppcRecoYuvTemp[uiSrcDepth]->DefaultConvertPix( cuCompRect.x0, cuCompRect.y0, cuCompRect.width, pcCU->getSlice()->getSPS()->getBitDepths() );
    m_ppcRecoYuvTemp[uiSrcDepth]->copyToPicYuv( rpcPic->getPicYuvCSC(), uiCUAddr, uiAbsPartIdx, uiDepth - uiSrcDepth, uiPartIdx);
  }

  m_ppcPredYuvBest[uiSrcDepth]->copyToPicYuv( rpcPic->getPicYuvPred (), uiCUAddr, uiAbsPartIdx, uiDepth - uiSrcDepth, uiPartIdx);
}

Void TEncCu::xCopyYuv2Tmp( UInt uiPartUnitIdx, UInt uiNextDepth )
{
  UInt uiCurrDepth = uiNextDepth - 1;
  m_ppcRecoYuvBest[uiNextDepth]->copyToPartYuv( m_ppcRecoYuvTemp[uiCurrDepth], uiPartUnitIdx );
  m_ppcPredYuvBest[uiNextDepth]->copyToPartYuv( m_ppcPredYuvBest[uiCurrDepth], uiPartUnitIdx);
}

/** Function for filling the PCM buffer of a CU using its original sample array
 * \param pCU pointer to current CU
 * \param pOrgYuv pointer to original sample array
 */
Void TEncCu::xFillPCMBuffer     ( TComDataCU* pCU, TComYuv* pOrgYuv )
{
  const ChromaFormat format = pCU->getPic()->getChromaFormat();
  const UInt numberValidComponents = getNumberValidComponents(format);
  for (UInt componentIndex = 0; componentIndex < numberValidComponents; componentIndex++)
  {
    const ComponentID component = ComponentID(componentIndex);

    const UInt width  = pCU->getWidth(0)  >> getComponentScaleX(component, format);
    const UInt height = pCU->getHeight(0) >> getComponentScaleY(component, format);

    Pel *source      = pOrgYuv->getAddr(component, 0, width);
    Pel *destination = pCU->getPCMSample(component);

    const UInt sourceStride = pOrgYuv->getStride(component);

    for (Int line = 0; line < height; line++)
    {
      for (Int column = 0; column < width; column++)
      {
        destination[column] = source[column];
      }

      source      += sourceStride;
      destination += width;
    }
  }
}

#if ADAPTIVE_QP_SELECTION
/** Collect ARL statistics from one block
  */
Int TEncCu::xTuCollectARLStats(TCoeff* rpcCoeff, TCoeff* rpcArlCoeff, Int NumCoeffInCU, Double* cSum, UInt* numSamples )
{
  for( Int n = 0; n < NumCoeffInCU; n++ )
  {
    TCoeff u = abs( rpcCoeff[ n ] );
    TCoeff absc = rpcArlCoeff[ n ];

    if( u != 0 )
    {
      if( u < LEVEL_RANGE )
      {
        cSum[ u ] += ( Double )absc;
        numSamples[ u ]++;
      }
      else
      {
        cSum[ LEVEL_RANGE ] += ( Double )absc - ( Double )( u << ARL_C_PRECISION );
        numSamples[ LEVEL_RANGE ]++;
      }
    }
  }

  return 0;
}

//! Collect ARL statistics from one CTU
Void TEncCu::xCtuCollectARLStats(TComDataCU* pCtu )
{
  Double cSum[ LEVEL_RANGE + 1 ];     //: the sum of DCT coefficients corresponding to data type and quantization output
  UInt numSamples[ LEVEL_RANGE + 1 ]; //: the number of coefficients corresponding to data type and quantization output

  TCoeff* pCoeffY = pCtu->getCoeff(COMPONENT_Y);
  TCoeff* pArlCoeffY = pCtu->getArlCoeff(COMPONENT_Y);
  const TComSPS &sps = *(pCtu->getSlice()->getSPS());

  const UInt uiMinCUWidth = sps.getMaxCUWidth() >> sps.getMaxTotalCUDepth(); // NOTE: ed - this is not the minimum CU width. It is the square-root of the number of coefficients per part.
  const UInt uiMinNumCoeffInCU = 1 << uiMinCUWidth;                          // NOTE: ed - what is this?

  memset( cSum, 0, sizeof( Double )*(LEVEL_RANGE+1) );
  memset( numSamples, 0, sizeof( UInt )*(LEVEL_RANGE+1) );

  // Collect stats to cSum[][] and numSamples[][]
  for(Int i = 0; i < pCtu->getTotalNumPart(); i ++ )
  {
    UInt uiTrIdx = pCtu->getTransformIdx(i);

    if(pCtu->isInter(i) && pCtu->getCbf( i, COMPONENT_Y, uiTrIdx ) )
    {
      xTuCollectARLStats(pCoeffY, pArlCoeffY, uiMinNumCoeffInCU, cSum, numSamples);
    }//Note that only InterY is processed. QP rounding is based on InterY data only.

    pCoeffY  += uiMinNumCoeffInCU;
    pArlCoeffY  += uiMinNumCoeffInCU;
  }

  for(Int u=1; u<LEVEL_RANGE;u++)
  {
    m_pcTrQuant->getSliceSumC()[u] += cSum[ u ] ;
    m_pcTrQuant->getSliceNSamples()[u] += numSamples[ u ] ;
  }
  m_pcTrQuant->getSliceSumC()[LEVEL_RANGE] += cSum[ LEVEL_RANGE ] ;
  m_pcTrQuant->getSliceNSamples()[LEVEL_RANGE] += numSamples[ LEVEL_RANGE ] ;
}
#endif

// SCM new added functions
// ====================================================================================================================
// Static function
// ====================================================================================================================
/** Calculates the minimum of the H and V luma activity of a CU in the source image.
*\param   pcCU
*\param   uiAbsPartIdx
*\param   ppcOrigYuv
*\returns Intermediate_Int
*
*/
static Intermediate_Int
CalculateMinimumHVLumaActivity(TComDataCU *pcCU, const UInt uiAbsPartIdx, const TComYuv * const * const ppcOrigYuv)
{
  const TComYuv *pOrgYuv = ppcOrigYuv[pcCU->getDepth(uiAbsPartIdx)];
  const Int      stride  = pOrgYuv->getStride(COMPONENT_Y);
  const Int      width   = pcCU->getWidth(uiAbsPartIdx);
  const Int      height  = pcCU->getHeight(uiAbsPartIdx);

  // Get activity
  Intermediate_Int hAct = 0;
  const Pel *pY = pOrgYuv->getAddr(COMPONENT_Y, uiAbsPartIdx);
  for (Int y = 0; y < height; y++)
  {
    for (Int x = 1; x < width; x++)
    {
      hAct += abs( pY[x] - pY[x - 1]);
    }
    pY += stride;
  }

  Intermediate_Int vAct = 0;
  pY = pOrgYuv->getAddr(COMPONENT_Y, 0) + stride;
  for (Int y = 1; y < height; y++)
  {
    for (Int x = 0; x < width; x++)
    {
      vAct += abs(pY[x] - pY[x - stride]);
    }
    pY += stride;
  }

  return std::min(hAct, vAct);
}

Void TEncCu::setEnableIntraTUACT(UInt depth, TComSlice* pcSlice)
{
  if( m_pcEncCfg->getRGBFormatFlag() )
  {
    m_bEnableIntraTUACTRD = true;
    if( (m_pcEncCfg->getGOPSize() == 8 && pcSlice->getDepth() == 3) || (m_pcEncCfg->getGOPSize() == 4 && pcSlice->getDepth() != 2) || (depth < 3) )
    {
      m_bEnableIntraTUACTRD = false;
    }
  }
  else
  {
    m_bEnableIntraTUACTRD = false;
  }
}

Void TEncCu::setEnableIBCTUACT(UInt depth, TComSlice* pcSlice)
{
  if( m_pcEncCfg->getRGBFormatFlag() )
  {
    m_bEnableIBCTUACTRD = true;
    if( (m_pcEncCfg->getGOPSize() == 8 && pcSlice->getDepth() == 3) || (m_pcEncCfg->getGOPSize() == 4 && pcSlice->getDepth() != 2) || (depth < 3) )
    {
      m_bEnableIBCTUACTRD = false;
    }
  }
  else
  {
    m_bEnableIBCTUACTRD = false;
  }
}

Void TEncCu::setEnableInterTUACT(UInt depth, TComSlice* pcSlice)
{
  if( m_pcEncCfg->getRGBFormatFlag() )
  {
    m_bEnableInterTUACTRD = true;
    if( (pcSlice->getDepth() == 0) || (depth < 2) )
    {
      m_bEnableInterTUACTRD = false;
    }
  }
  else
  {
    m_bEnableInterTUACTRD = false;
  }
}

Void TEncCu::xCheckRDCostIntraBCMerge2Nx2N( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU )
{
  TComMvField  cMvFieldNeighbours[2 * MRG_MAX_NUM_CANDS]; // double length for mv of both lists
  UChar interDirNeighbours[MRG_MAX_NUM_CANDS];
  Int numValidMergeCand = 0;
  const Bool bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);

  for( UInt ui = 0; ui < rpcTempCU->getSlice()->getMaxNumMergeCand(); ++ui )
  {
    interDirNeighbours[ui] = 0;
  }

  Int xPos = rpcTempCU->getCUPelX();
  Int yPos = rpcTempCU->getCUPelY();
  Int width = rpcTempCU->getWidth( 0 );
  Int height = rpcTempCU->getHeight( 0 );
  UChar depth = rpcTempCU->getDepth( 0 );
  rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth );
#if MCTS_ENC_CHECK
  UInt numSpatialMergeCandidates = 0;
  rpcTempCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours,interDirNeighbours, numValidMergeCand, numSpatialMergeCandidates );
#else
  rpcTempCU->getInterMergeCandidates( 0, 0, cMvFieldNeighbours,interDirNeighbours, numValidMergeCand );
#endif
  rpcTempCU->roundMergeCandidates(cMvFieldNeighbours, numValidMergeCand);
  rpcTempCU->xRestrictBipredMergeCand( 0, cMvFieldNeighbours, interDirNeighbours, numValidMergeCand );

  Int mergeCandBuffer[MRG_MAX_NUM_CANDS];
  for( UInt ui = 0; ui < numValidMergeCand; ++ui )
  {
    mergeCandBuffer[ui] = 0;
  }

#if MCTS_ENC_CHECK
  if (m_pcEncCfg->getTMCTSSEITileConstraint() && rpcTempCU->isLastColumnCTUInTile())
  {
    numValidMergeCand = numSpatialMergeCandidates;
  }
#endif

  Bool bestIsSkip = false;

  UInt iteration;
  if ( rpcTempCU->isLosslessCoded(0))
  {
    iteration = 1;
  }
  else
  {
    iteration = 2;
  }

  const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
  for( UInt noResidual = 0; noResidual < iteration; ++noResidual )
  {
    for( UInt mergeCand = 0; mergeCand < numValidMergeCand; ++mergeCand )
    {
      if(!(noResidual==1 && mergeCandBuffer[mergeCand]==1))
      {
        if( !(bestIsSkip && noResidual == 0) )
        {
          if ( interDirNeighbours[mergeCand] != 1 )
          {
            continue;
          }

          if ( rpcTempCU->getSlice()->getRefPic( REF_PIC_LIST_0, cMvFieldNeighbours[mergeCand<<1].getRefIdx() )->getPOC() != rpcTempCU->getSlice()->getPOC() )
          {
            continue;
          }

          if ( !m_pcPredSearch->isBlockVectorValid( xPos, yPos, width, height, rpcTempCU,
            0, 0, (cMvFieldNeighbours[mergeCand<<1].getHor() >> 2), (cMvFieldNeighbours[mergeCand<<1].getVer()>>2), sps.getMaxCUWidth() ) )
          {
            continue;
          }

          // set MC parameters
          rpcTempCU->setPredModeSubParts( MODE_INTER, 0, depth ); // interprets depth relative to LCU level
          rpcTempCU->setCUTransquantBypassSubParts( bTransquantBypassFlag, 0, depth );
          rpcTempCU->setChromaQpAdjSubParts( bTransquantBypassFlag ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, depth );
          rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth ); // interprets depth relative to LCU level
          rpcTempCU->setMergeFlagSubParts( true, 0, 0, depth ); // interprets depth relative to LCU level
          rpcTempCU->setMergeIndexSubParts( mergeCand, 0, 0, depth ); // interprets depth relative to LCU level
          rpcTempCU->setInterDirSubParts( interDirNeighbours[mergeCand], 0, 0, depth ); // interprets depth relative to LCU level
          rpcTempCU->getCUMvField( REF_PIC_LIST_0 )->setAllMvField( cMvFieldNeighbours[0 + 2*mergeCand], SIZE_2Nx2N, 0, 0 ); // interprets depth relative to rpcTempCU level
          rpcTempCU->getCUMvField( REF_PIC_LIST_1 )->setAllMvField( cMvFieldNeighbours[1 + 2*mergeCand], SIZE_2Nx2N, 0, 0 ); // interprets depth relative to rpcTempCU level

#if MCTS_ENC_CHECK
          if ( m_pcEncCfg->getTMCTSSEITileConstraint () && (!(m_pcPredSearch->checkTMctsMvp(rpcTempCU))))
          {
            continue;
          }
#endif

          // do MC
          m_pcPredSearch->motionCompensation ( rpcTempCU, m_ppcPredYuvTemp[depth] );
          Bool bColourTrans = (m_pcEncCfg->getRGBFormatFlag() && rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans())? true : false;

          if( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
          {
            bColourTrans = false;
          }

          TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[depth];
          if( rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && (!bTransquantBypassFlag || sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
          {
            if(!getEnableIBCTUACT())
            {
              m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                         m_ppcOrigYuv    [depth],
                                                         pcTmpPredYuv           ,
                                                         m_ppcResiYuvTemp[depth],
                                                         m_ppcResiYuvBest[depth],
                                                         m_ppcRecoYuvTemp[depth],
                                                         (noResidual != 0),
                                                         m_ppcNoCorrYuv  [depth],
                                                         (bColourTrans? ACT_TRAN_CLR: ACT_ORG_CLR)
                                                         DEBUG_STRING_PASS_INTO(tmpStr) );
            }
            else
            {
              m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                         m_ppcOrigYuv    [depth],
                                                         pcTmpPredYuv           ,
                                                         m_ppcResiYuvTemp[depth],
                                                         m_ppcResiYuvBest[depth],
                                                         m_ppcRecoYuvTemp[depth],
                                                         (noResidual != 0),
                                                         m_ppcNoCorrYuv  [depth],
                                                         ACT_TWO_CLR
                                                         DEBUG_STRING_PASS_INTO(tmpStr) );
            }
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                       m_ppcOrigYuv    [depth],
                                                       pcTmpPredYuv           ,
                                                       m_ppcResiYuvTemp[depth],
                                                       m_ppcResiYuvBest[depth],
                                                       m_ppcRecoYuvTemp[depth],
                                                       (noResidual != 0),
                                                       m_ppcNoCorrYuv  [depth],
                                                       ACT_ORG_CLR
                                                       DEBUG_STRING_PASS_INTO(tmpStr) );
          }

          if ((noResidual == 0) && (rpcTempCU->getQtRootCbf(0) == 0))
          {
            // If no residual when allowing for one, then set mark to not try case where residual is forced to 0
            mergeCandBuffer[mergeCand] = 1;
          }

          rpcTempCU->setSkipFlagSubParts( rpcTempCU->getQtRootCbf(0) == 0, 0, depth );
          Int orgQP = rpcTempCU->getQP( 0 );
          xCheckDQP( rpcTempCU );
          TComDataCU *rpcTempCUPre = rpcTempCU;
          xCheckBestMode(rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(bestStr) DEBUG_STRING_PASS_INTO(tmpStr));

          if(rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && !noResidual && rpcTempCUPre->getQtRootCbf(0) )
          {
            if( rpcTempCU != rpcTempCUPre )
            {
              rpcTempCU->initEstData( depth, orgQP, bTransquantBypassFlag );
              rpcTempCU->copyPartFrom( rpcBestCU, 0, depth );
            }
            if(!getEnableIBCTUACT())
            {
              bColourTrans = !bColourTrans;
            }
            else
            {
              bColourTrans = m_pcEncCfg->getRGBFormatFlag();
            }
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU,
                                                       m_ppcOrigYuv    [depth],
                                                       pcTmpPredYuv           ,
                                                       m_ppcResiYuvTemp[depth],
                                                       m_ppcResiYuvBest[depth],
                                                       m_ppcRecoYuvTemp[depth],
                                                       (noResidual != 0),
                                                       m_ppcNoCorrYuv  [depth],
                                                       (bColourTrans? ACT_TRAN_CLR: ACT_ORG_CLR)
                                                       DEBUG_STRING_PASS_INTO(tmpStr) );
            rpcTempCU->setSkipFlagSubParts( rpcTempCU->getQtRootCbf(0) == 0, 0, depth );
            //Double rdCost = rpcTempCU->getTotalCost();

            xCheckDQP( rpcTempCU );
            if(rpcTempCU->getQtRootCbf(0))
            {
              xCheckBestMode( rpcBestCU, rpcTempCU, depth );
            }
          }
          rpcTempCU->initEstData( depth, orgQP, bTransquantBypassFlag );

          if( m_pcEncCfg->getUseFastDecisionForMerge() && !bestIsSkip )
          {
            bestIsSkip = rpcBestCU->getQtRootCbf(0) == 0;
          }
        }
      }
    }
  }
}

Void TEncCu::xCheckRDCostIntraCSC( TComDataCU     *&rpcBestCU,
                                   TComDataCU     *&rpcTempCU,
                                    Double         &cost,
                                   PartSize       eSize,
                                   ACTRDTestTypes eACTRDTestType,
                                   Bool           bReuseIntraMode
                                   DEBUG_STRING_FN_DECLARE(sDebug)
)
{
  DEBUG_STRING_NEW(sTest)

  UInt depth = rpcTempCU->getDepth( 0 );
  rpcTempCU->setSkipFlagSubParts( false, 0, depth );
  rpcTempCU->setPartSizeSubParts( eSize, 0, depth );
  rpcTempCU->setPredModeSubParts( MODE_INTRA, 0, depth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, depth );

  m_pcPredSearch->estIntraPredQTCT( rpcTempCU, m_ppcOrigYuv[depth], m_ppcPredYuvTemp[depth], m_ppcResiYuvTemp[depth], m_ppcRecoYuvTemp[depth], eACTRDTestType, bReuseIntraMode DEBUG_STRING_PASS_INTO(sTest) );

  m_pcRDGoOnSbacCoder->load(m_pppcRDSbacCoder[depth][CI_CURR_BEST]);
  m_pcEntropyCoder->resetBits();

  if ( rpcTempCU->getSlice()->getPPS()->getTransquantBypassEnabledFlag() )
  {
    m_pcEntropyCoder->encodeCUTransquantBypassFlag( rpcTempCU, 0, true );
  }

  m_pcEntropyCoder->encodeSkipFlag ( rpcTempCU, 0, true );

  m_pcEntropyCoder->encodePredMode( rpcTempCU, 0, true );

  Bool bCodeDQP = getdQPFlag();
  Bool codeChromaQpAdjFlag = getCodeChromaQpAdjFlag();

  m_pcEntropyCoder->encodePaletteModeInfo( rpcTempCU, 0, true, &bCodeDQP, &codeChromaQpAdjFlag );

  m_pcEntropyCoder->encodePartSize( rpcTempCU, 0, depth, true );
  m_pcEntropyCoder->encodePredInfo( rpcTempCU, 0 );
  m_pcEntropyCoder->encodeIPCMInfo(rpcTempCU, 0, true );

  // Encode Coefficients
  m_pcEntropyCoder->encodeCoeff( rpcTempCU, 0, depth, bCodeDQP, codeChromaQpAdjFlag );
  setCodeChromaQpAdjFlag( codeChromaQpAdjFlag );
  setdQPFlag( bCodeDQP );

  m_pcRDGoOnSbacCoder->store(m_pppcRDSbacCoder[depth][CI_TEMP_BEST]);

  rpcTempCU->getTotalBits() = m_pcEntropyCoder->getNumberOfWrittenBits();
  rpcTempCU->getTotalBins() = ((TEncBinCABAC *)((TEncSbac*)m_pcEntropyCoder->m_pcEntropyCoderIf)->getEncBinIf())->getBinsCoded();
  rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

  xCheckDQP( rpcTempCU );

  cost = rpcTempCU->getTotalCost();

  xCheckBestMode(rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
}

Void TEncCu::xCheckRDCostIntraBC( TComDataCU *&rpcBestCU,
                                  TComDataCU *&rpcTempCU,
                                  Bool         bUse1DSearchFor8x8,
                                  PartSize     eSize,
                                  Double      &rdCost,
                                  Bool         testPredOnly,
                                  TComMv      *iMVCandList
                                  DEBUG_STRING_FN_DECLARE(sDebug))
{
  DEBUG_STRING_NEW(sTest)
  UInt depth = rpcTempCU->getDepth( 0 );

  rpcTempCU->setDepthSubParts( depth, 0 );
  rpcTempCU->setSkipFlagSubParts( false, 0, depth );
  rpcTempCU->setPartSizeSubParts( eSize, 0, depth );
  rpcTempCU->setPredModeSubParts( MODE_INTER, 0, depth );

  rpcTempCU->setIntraDirSubParts( CHANNEL_TYPE_LUMA, DC_IDX, 0, depth );
  rpcTempCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, DC_IDX, 0, depth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, depth );

#if MCTS_ENC_CHECK
  rpcTempCU->setTMctsMvpIsValid(true);
#endif

  // intra BV search
  const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
  if( rpcTempCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans () && m_pcEncCfg->getRGBFormatFlag() && (!rpcTempCU->getCUTransquantBypass(0) || (sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA ))) )
  {
    m_ppcOrigYuv[depth]->copyFromPicYuv( rpcTempCU->getPic()->getPicYuvResi(), rpcTempCU->getCtuRsAddr(), rpcTempCU->getZorderIdxInCtu() );
    rpcTempCU->getPic()->exchangePicYuvRec();  //YCgCo reference picture
  }

  Bool bValid = m_pcPredSearch->predIntraBCSearch ( rpcTempCU,
                                                    m_ppcOrigYuv[depth],
                                                    m_ppcPredYuvTemp[depth],
                                                    m_ppcResiYuvTemp[depth],
                                                    m_ppcRecoYuvTemp[depth]
                                                    DEBUG_STRING_PASS_INTO(sTest),
                                                    bUse1DSearchFor8x8,
                                                    false,
                                                    testPredOnly
                                                  );

  if ( bValid && (rpcTempCU->getWidth( 0 ) <= 16) && (eSize == SIZE_2NxN || eSize == SIZE_Nx2N) )
  {
    Int dummyWidth, dummyHeight;
    UInt partAddr = 0;
    rpcTempCU->getPartIndexAndSize( 1, partAddr, dummyWidth, dummyHeight );
    iMVCandList[0] = rpcTempCU->getCUMvField( REF_PIC_LIST_0 )->getMv( 0 );
    iMVCandList[1] = rpcTempCU->getCUMvField( REF_PIC_LIST_0 )->getMv( partAddr );
  }
  if( rpcTempCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans() && m_pcEncCfg->getRGBFormatFlag() && (!rpcTempCU->getCUTransquantBypass(0) || (sps.getBitDepth( CHANNEL_TYPE_LUMA ) == sps.getBitDepth( CHANNEL_TYPE_CHROMA ))) )
  {
    m_ppcOrigYuv[depth]->copyFromPicYuv( rpcTempCU->getPic()->getPicYuvOrg(), rpcTempCU->getCtuRsAddr(), rpcTempCU->getZorderIdxInCtu() );
    rpcTempCU->getPic()->exchangePicYuvRec();
  }
  
  if (bValid)
  {
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!rpcTempCU->getTMctsMvpIsValid()))
    {
      return;
    }
#endif
    TComDataCU *rpcTempCUPre = NULL;
    Bool   bEnableTrans      = rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans();
    UChar  colourTransform = 0;
    Bool   bRGB              = m_pcEncCfg->getRGBFormatFlag();
    Double dCostFst          = MAX_DOUBLE;
    SChar   orgQP            = rpcTempCU->getQP( 0 );
    Bool   bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);
    if ( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
    {
      bEnableTrans = false;
    }
    TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[depth];
    for(UInt i = 0;  i < (testPredOnly ? 1 : 2) ; i++)
    {
      colourTransform = ( bRGB && bEnableTrans )? (1-i): i;

      if( colourTransform && m_pcEncCfg->getRGBFormatFlag())
      {
        m_pcPredSearch->motionCompensation ( rpcTempCU, m_ppcPredYuvTemp[depth] );
      }

      if ( i == 0 )
      {
        if ( bEnableTrans )
        {
          if ( !getEnableIBCTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], (colourTransform? ACT_TRAN_CLR: ACT_ORG_CLR) DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TWO_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
      else
      {
        if ( m_pcEncCfg->getRGBFormatFlag() )
        {
          if ( !getEnableIBCTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          if ( !getEnableIBCTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
      }
      rpcTempCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );
      rdCost = rpcTempCU->getTotalCost();

#if DEBUG_STRING
      DebugInterPredResiReco(sTest, *(m_ppcPredYuvTemp[depth]), *(m_ppcResiYuvBest[depth]), *(m_ppcRecoYuvTemp[depth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif

      if ( bEnableTrans )
      {
        if ( !i )
        {
          if ( rdCost < m_ppcBestCU[depth]->getTmpIntraBCRDCost() )
          {
            m_ppcBestCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcBestCU[depth]->setIntraBCCSCEnabled(true);
            m_ppcTempCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcTempCU[depth]->setIntraBCCSCEnabled(true);
          }
        }
        else
        {
          if ( rdCost < m_ppcBestCU[depth]->getTmpIntraBCRDCost() )
          {
            m_ppcBestCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcBestCU[depth]->setIntraBCCSCEnabled(false);
            m_ppcTempCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcTempCU[depth]->setIntraBCCSCEnabled(false);
          }
        }
      }
      xCheckDQP( rpcTempCU );
      rpcTempCUPre = rpcTempCU;
      xCheckBestMode(rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
      if( bEnableTrans )
      {
        if( testPredOnly || !rpcBestCU->isIntraBC(0) )
        {
          return;
        }
        Bool bParentUseCSC = ( ((depth > 2)) && (m_ppcBestCU[depth -1]->getIntraBCCSCEnabled()) );
        if(!i && (!rpcTempCUPre->getQtRootCbf(0) || bParentUseCSC))
        {
          return;
        }
        else if(!i)
        {
          dCostFst = rdCost;
          if( rpcTempCU != rpcTempCUPre )
          {
            rpcTempCU->initEstData( depth, orgQP, bTransquantBypassFlag );
            rpcTempCU->copyPartFrom( rpcBestCU, 0, depth );
          }
        }
        else
        {
          if ( rpcTempCU == rpcTempCUPre )
          {
            rdCost = dCostFst;
          }
        }
      }
      else
      {
        return;
      }
    }
  }
  else
  {
    rdCost = MAX_DOUBLE;
  }
}

Void TEncCu::xCheckRDCostIntraBCMixed( TComDataCU *&rpcBestCU,
                                       TComDataCU *&rpcTempCU,
                                       PartSize     eSize,
                                       Double      &rdCost
                                       DEBUG_STRING_FN_DECLARE( sDebug ),
                                       TComMv* iMVCandList
                                     )
{
  DEBUG_STRING_NEW( sTest )
  UInt depth = rpcTempCU->getDepth( 0 );
  rpcTempCU->setPredModeSubParts( MODE_INTER, 0, depth );
  rpcTempCU->setDepthSubParts( depth, 0 );
  rpcTempCU->setSkipFlagSubParts( false, 0, depth );
  rpcTempCU->setPartSizeSubParts( eSize, 0, depth );
  rpcTempCU->setIntraDirSubParts( CHANNEL_TYPE_LUMA, DC_IDX, 0, depth );
  rpcTempCU->setIntraDirSubParts( CHANNEL_TYPE_CHROMA, DC_IDX, 0, depth );
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass( 0 ) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, depth );

#if MCTS_ENC_CHECK
  rpcTempCU->setTMctsMvpIsValid(true);
#endif

  Bool bValid = m_pcPredSearch->predMixedIntraBCInterSearch( rpcTempCU,
                                                             m_ppcOrigYuv[depth],
                                                             m_ppcPredYuvTemp[depth],
                                                             m_ppcResiYuvTemp[depth],
                                                             m_ppcRecoYuvTemp[depth]
                                                             DEBUG_STRING_PASS_INTO( sTest ),
                                                             iMVCandList,
                                                             false
                                                           );

  if ( bValid )
  {
#if MCTS_ENC_CHECK
    if (m_pcEncCfg->getTMCTSSEITileConstraint() && (!rpcTempCU->getTMctsMvpIsValid()))
    {
      return;
    }
#endif
    TComDataCU *rpcTempCUPre = NULL;
    Bool   bEnableTrans      = rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans();
    UChar  colourTransform = 0;
    Bool   bRGB              = m_pcEncCfg->getRGBFormatFlag();
    Double dCostFst          = MAX_DOUBLE;
    SChar   orgQP            = rpcTempCU->getQP( 0 );
    Bool   bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);
    const TComSPS &sps       = *(rpcTempCU->getSlice()->getSPS());
    if ( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
    {
      bEnableTrans = false;
    }
    TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[depth];

    for(UInt i = 0;  i < 2 ; i++)
    {
      colourTransform = (bRGB && bEnableTrans)? (1-i): i;
      if ( i == 0 )
      {
        if ( bEnableTrans )
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], (colourTransform? ACT_TRAN_CLR: ACT_ORG_CLR) DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TWO_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
      else
      {
        if ( m_pcEncCfg->getRGBFormatFlag() )
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
      }
      rpcTempCU->getTotalCost()  = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );
      rdCost = rpcTempCU->getTotalCost();

#if DEBUG_STRING
      DebugInterPredResiReco(sTest, *(m_ppcPredYuvTemp[depth]), *(m_ppcResiYuvBest[depth]), *(m_ppcRecoYuvTemp[depth]), DebugStringGetPredModeMask(rpcTempCU->getPredictionMode(0)));
#endif
      if ( bEnableTrans )
      {
        if ( !i )
        {
          if ( rdCost < m_ppcBestCU[depth]->getTmpIntraBCRDCost() )
          {
            m_ppcBestCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcBestCU[depth]->setIntraBCCSCEnabled(true);
            m_ppcTempCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcTempCU[depth]->setIntraBCCSCEnabled(true);
          }
        }
        else
        {
          if ( rdCost < m_ppcBestCU[depth]->getTmpIntraBCRDCost() )
          {
            m_ppcBestCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcBestCU[depth]->setIntraBCCSCEnabled(false);
            m_ppcTempCU[depth]->setTmpIntraBCRDCost(rdCost);
            m_ppcTempCU[depth]->setIntraBCCSCEnabled(false);
          }
        }
      }
      xCheckDQP( rpcTempCU );
      rpcTempCUPre = rpcTempCU;
      xCheckBestMode(rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest));
      if( bEnableTrans )
      {
        Bool bParentUseCSC = ( ((depth > 2)) && (m_ppcBestCU[depth -1]->getIntraBCCSCEnabled()) );
        if(!i && (!rpcTempCUPre->getQtRootCbf(0) || bParentUseCSC))
        {
          return;
        }
        else if(!i)
        {
          dCostFst = rdCost;
          if( rpcTempCU != rpcTempCUPre )
          {
            rpcTempCU->initEstData( depth, orgQP, bTransquantBypassFlag );
            rpcTempCU->copyPartFrom( rpcBestCU, 0, depth );
          }
        }
        else
        {
          if ( rpcTempCU == rpcTempCUPre )
          {
            rdCost = dCostFst;
          }
        }
      }
      else
      {
        return;
      }
    }
  }
  else
  {
    rdCost = MAX_DOUBLE;
  }
}

Void TEncCu::xCheckRDCostHashInter( TComDataCU*& rpcBestCU, TComDataCU*& rpcTempCU, Bool& isPerfectMatch DEBUG_STRING_FN_DECLARE(sDebug) )
{
  DEBUG_STRING_NEW(sTest)

  isPerfectMatch = false;
  UChar depth = rpcTempCU->getDepth( 0 );
  rpcTempCU->setDepthSubParts( depth, 0 );
  rpcTempCU->setSkipFlagSubParts( false, 0, depth );
  rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth );
  rpcTempCU->setPredModeSubParts( MODE_INTER, 0, depth );
  rpcTempCU->setMergeFlagSubParts( false, 0, 0, depth );

  if ( m_pcPredSearch->predInterHashSearch( rpcTempCU, m_ppcPredYuvTemp[depth], isPerfectMatch ) )
  {
    TComDataCU *rpcTempCUPre = NULL;
    Bool   bEnableTrans      = rpcBestCU->getSlice()->getPPS()->getPpsScreenExtension().getUseColourTrans();
    UChar  colourTransform = 0;
    Bool   bRGB              = m_pcEncCfg->getRGBFormatFlag();
    SChar   orgQP            = rpcTempCU->getQP( 0 );
    Bool   bTransquantBypassFlag = rpcTempCU->getCUTransquantBypass(0);
    const TComSPS &sps=*(rpcTempCU->getSlice()->getSPS());
    if ( bTransquantBypassFlag && (sps.getBitDepth( CHANNEL_TYPE_LUMA ) != sps.getBitDepth( CHANNEL_TYPE_CHROMA )) )
    {
      bEnableTrans = false;
    }

    TComYuv* pcTmpPredYuv = m_ppcPredYuvTemp[depth];
    for(UInt i = 0;  i < 2 ; i++)
    {
      colourTransform = ( bRGB && bEnableTrans )? (1-i): i;
      if ( i == 0 )
      {
        if ( bEnableTrans )
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], (colourTransform? ACT_TRAN_CLR: ACT_ORG_CLR) DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TWO_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
        }
      }
      else
      {
        if ( m_pcEncCfg->getRGBFormatFlag() )
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
        else
        {
          if ( !getEnableInterTUACT() )
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_TRAN_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
          else
          {
            m_pcPredSearch->encodeResAndCalcRdInterCU( rpcTempCU, m_ppcOrigYuv[depth], pcTmpPredYuv, m_ppcResiYuvTemp[depth], m_ppcResiYuvBest[depth], m_ppcRecoYuvTemp[depth], false, m_ppcNoCorrYuv[depth], ACT_ORG_CLR DEBUG_STRING_PASS_INTO(sTest) );
          }
        }
      }
      rpcTempCU->getTotalCost() = m_pcRdCost->calcRdCost( rpcTempCU->getTotalBits(), rpcTempCU->getTotalDistortion() );

      if(!i)
      {
        Double rdCost1 = rpcTempCU->getTotalCost();
        if( rdCost1 < m_ppcBestCU[depth]->getTmpInterRDCost() )
        {
          m_ppcBestCU[depth]->setTmpInterRDCost(rdCost1);
          m_ppcBestCU[depth]->setInterCSCEnabled(true);
          m_ppcTempCU[depth]->setTmpInterRDCost(rdCost1);
          m_ppcTempCU[depth]->setInterCSCEnabled(true);
        }
      }
      else
      {
        Double rdCost = rpcTempCU->getTotalCost();
        if( rdCost < m_ppcBestCU[depth]->getTmpInterRDCost() )
        {
          m_ppcBestCU[depth]->setTmpInterRDCost(rdCost);
          m_ppcBestCU[depth]->setInterCSCEnabled(false);
          m_ppcTempCU[depth]->setTmpInterRDCost(rdCost);
          m_ppcTempCU[depth]->setInterCSCEnabled(false);
        }
      }

      xCheckDQP( rpcTempCU );
      rpcTempCUPre = rpcTempCU;

      xCheckBestMode( rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(sDebug) DEBUG_STRING_PASS_INTO(sTest) );
      if(bEnableTrans)
      {
        if( !rpcBestCU->isInter(0) || rpcBestCU->isSkipped(0)  )
        {
          return;
        }
        Bool bParentUseCSC = ( (depth != 0) && (m_ppcBestCU[depth -1]->getInterCSCEnabled()) );
        if(!i && (!rpcTempCUPre->getQtRootCbf(0) || bParentUseCSC))
        {
          return;
        }
        else if(!i && rpcTempCUPre->getQtRootCbf(0))
        {
          if( rpcTempCU != rpcTempCUPre )
          {
            rpcTempCU->initEstData( depth, orgQP, bTransquantBypassFlag);
            rpcTempCU->copyPartFrom( rpcBestCU, 0, depth );
          }
        }
      }
      else
      {
        return;
      }
    }
  }
}

UInt TEncCu::xCheckPaletteMode(TComDataCU *&rpcBestCU, TComDataCU *&rpcTempCU, Bool forcePalettePrediction, UInt iterNumber, UInt *paletteSize)
{
  UInt depth = rpcTempCU->getDepth( 0 );

  rpcTempCU->setSkipFlagSubParts( false, 0, depth );
  rpcTempCU->setIPCMFlag(0, false);
  rpcTempCU->setIPCMFlagSubParts (false, 0, rpcTempCU->getDepth(0));
  rpcTempCU->setPartSizeSubParts( SIZE_2Nx2N, 0, depth );
  rpcTempCU->setPredModeSubParts( MODE_INTRA, 0, depth );
  rpcTempCU->setTrIdxSubParts ( 0, 0, depth );
  rpcTempCU->setPaletteModeFlagSubParts(true, 0, rpcTempCU->getDepth(0));
  rpcTempCU->setChromaQpAdjSubParts( rpcTempCU->getCUTransquantBypass(0) ? 0 : m_cuChromaQpOffsetIdxPlus1, 0, depth );

  UInt testedModes=m_pcPredSearch->paletteSearch(rpcTempCU, m_ppcOrigYuv[depth], m_ppcPredYuvTemp[depth], m_ppcResiYuvTemp[depth],
                                                 m_ppcRecoYuvTemp[depth], forcePalettePrediction, iterNumber, paletteSize);

  xCheckDQP( rpcTempCU );
  DEBUG_STRING_NEW(a)
  DEBUG_STRING_NEW(b)
  xCheckBestMode(rpcBestCU, rpcTempCU, depth DEBUG_STRING_PASS_INTO(a) DEBUG_STRING_PASS_INTO(b));

  return (testedModes);
}

//! \}
