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

/** \file     TDecSlice.cpp
    \brief    slice decoder class
*/

#include "TDecSlice.h"
#include "TDecConformance.h"

//! \ingroup TLibDecoder
//! \{

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

TDecSlice::TDecSlice()
{
}

TDecSlice::~TDecSlice()
{
}

Void TDecSlice::create()
{
}

Void TDecSlice::destroy()
{
}

Void TDecSlice::init(TDecEntropy* pcEntropyDecoder, TDecCu* pcCuDecoder, TDecConformanceCheck *pDecConformanceCheck)
{
  m_pcEntropyDecoder     = pcEntropyDecoder;
  m_pcCuDecoder          = pcCuDecoder;
  m_pDecConformanceCheck = pDecConformanceCheck;
}

Void TDecSlice::decompressSlice(TComInputBitstream** ppcSubstreams, TComPic* pcPic, TDecSbac* pcSbacDecoder)
{
  TComSlice* pcSlice                 = pcPic->getSlice(pcPic->getCurrSliceIdx());

  const Int  startCtuTsAddr          = pcSlice->getSliceSegmentCurStartCtuTsAddr();
  const Int  startCtuRsAddr          = pcPic->getPicSym()->getCtuTsToRsAddrMap(startCtuTsAddr);
  const UInt numCtusInFrame          = pcPic->getNumberOfCtusInFrame();

  const UInt frameWidthInCtus        = pcPic->getPicSym()->getFrameWidthInCtus();
  const Bool depSliceSegmentsEnabled = pcSlice->getPPS()->getDependentSliceSegmentsEnabledFlag();
  const Bool wavefrontsEnabled       = pcSlice->getPPS()->getEntropyCodingSyncEnabledFlag();

  m_pcEntropyDecoder->setEntropyDecoder ( pcSbacDecoder  );
  m_pcEntropyDecoder->setBitstream      ( ppcSubstreams[0] );
  m_pcEntropyDecoder->resetEntropy      (pcSlice);

  // decoder doesn't need prediction & residual frame buffer
  pcPic->setPicYuvPred( 0 );
  pcPic->setPicYuvResi( 0 );

#if ENC_DEC_TRACE
  g_bJustDoIt = g_bEncDecTraceEnable;
#endif
  DTRACE_CABAC_VL( g_nSymbolCounter++ );
  DTRACE_CABAC_T( "\tPOC: " );
  DTRACE_CABAC_V( pcPic->getPOC() );
  DTRACE_CABAC_T( "\n" );

#if ENC_DEC_TRACE
  g_bJustDoIt = g_bEncDecTraceDisable;
#endif

  UChar lastPaletteSize[MAX_NUM_COMPONENT] = { 0, 0, 0 };
  Pel lastPalette[MAX_NUM_COMPONENT][MAX_PALETTE_PRED_SIZE];
  for(UChar comp=0; comp<MAX_NUM_COMPONENT; comp++)
  {
    memset(lastPalette[comp], 0, sizeof(Pel) * pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize());
  }
  if (pcSlice->getPPS()->getPpsScreenExtension().getUsePalettePredictor())
  {
    xSetPredFromPPS(lastPalette, lastPaletteSize, pcSlice->getPPS(), pcSlice->getSPS());
  }
  else if (pcSlice->getSPS()->getSpsScreenExtension().getUsePalettePredictor())
  {
    xSetPredFromSPS(lastPalette, lastPaletteSize, pcSlice->getSPS());
  }
  else
  {
    xSetPredDefault(lastPalette, lastPaletteSize, pcSlice->getSPS());
  }

  // The first CTU of the slice is the first coded substream, but the global substream number, as calculated by getSubstreamForCtuAddr may be higher.
  // This calculates the common offset for all substreams in this slice.
  const UInt subStreamOffset=pcPic->getSubstreamForCtuAddr(startCtuRsAddr, true, pcSlice);


  if (depSliceSegmentsEnabled)
  {
    // modify initial contexts with previous slice segment if this is a dependent slice.
    const UInt startTileIdx=pcPic->getPicSym()->getTileIdxMap(startCtuRsAddr);
    const TComTile *pCurrentTile=pcPic->getPicSym()->getTComTile(startTileIdx);
    const UInt firstCtuRsAddrOfTile = pCurrentTile->getFirstCtuRsAddr();

    if( pcSlice->getDependentSliceSegmentFlag() && startCtuRsAddr != firstCtuRsAddrOfTile)
    {
      if ( pCurrentTile->getTileWidthInCtus() >= 2 || !wavefrontsEnabled)
      {
        pcSbacDecoder->loadContexts(&m_lastSliceSegmentEndContextState);
        for ( UChar comp = 0; comp < (pcSlice->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++ )
        {
          lastPaletteSize[comp] = m_lastSliceSegmentEndPaletteState.lastPaletteSize[comp];
          for ( UInt idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++ )
          {
            lastPalette[comp][idx] = m_lastSliceSegmentEndPaletteState.lastPalette[comp][idx];
          }
        }
      }
    }
  }

  // for every CTU in the slice segment...

  Bool isLastCtuOfSliceSegment = false;
  for( UInt ctuTsAddr = startCtuTsAddr; !isLastCtuOfSliceSegment && ctuTsAddr < numCtusInFrame; ctuTsAddr++)
  {
    const UInt ctuRsAddr = pcPic->getPicSym()->getCtuTsToRsAddrMap(ctuTsAddr);
    const TComTile &currentTile = *(pcPic->getPicSym()->getTComTile(pcPic->getPicSym()->getTileIdxMap(ctuRsAddr)));
    const UInt firstCtuRsAddrOfTile = currentTile.getFirstCtuRsAddr();
    const UInt tileXPosInCtus = firstCtuRsAddrOfTile % frameWidthInCtus;
    const UInt tileYPosInCtus = firstCtuRsAddrOfTile / frameWidthInCtus;
    const UInt ctuXPosInCtus  = ctuRsAddr % frameWidthInCtus;
    const UInt ctuYPosInCtus  = ctuRsAddr / frameWidthInCtus;
    const UInt uiSubStrm=pcPic->getSubstreamForCtuAddr(ctuRsAddr, true, pcSlice)-subStreamOffset;
    TComDataCU* pCtu = pcPic->getCtu( ctuRsAddr );
    pCtu->initCtu( pcPic, ctuRsAddr );

    m_pcEntropyDecoder->setBitstream( ppcSubstreams[uiSubStrm] );

    // set up CABAC contexts' state for this CTU
    if (ctuRsAddr == firstCtuRsAddrOfTile)
    {
      if (ctuTsAddr != startCtuTsAddr) // if it is the first CTU, then the entropy coder has already been reset
      {
        m_pcEntropyDecoder->resetEntropy(pcSlice);

        for (UChar comp = 0; comp<MAX_NUM_COMPONENT; comp++)
        {
          memset(lastPalette[comp], 0, sizeof(Pel) * pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize());
        }
        if (pcSlice->getPPS()->getPpsScreenExtension().getUsePalettePredictor())
        {
          xSetPredFromPPS(lastPalette, lastPaletteSize, pcSlice->getPPS(), pcSlice->getSPS());
        }
        else if (pcSlice->getSPS()->getSpsScreenExtension().getUsePalettePredictor())
        {
          xSetPredFromSPS(lastPalette, lastPaletteSize, pcSlice->getSPS());
        }
        else
        {
          xSetPredDefault(lastPalette, lastPaletteSize, pcSlice->getSPS());
        }
      }
    }
    else if (ctuXPosInCtus == tileXPosInCtus && wavefrontsEnabled)
    {
      // Synchronize cabac probabilities with upper-right CTU if it's available and at the start of a line.
      if (ctuTsAddr != startCtuTsAddr) // if it is the first CTU, then the entropy coder has already been reset
      {
        m_pcEntropyDecoder->resetEntropy(pcSlice);
      }
      TComDataCU *pCtuUp = pCtu->getCtuAbove();
      if ( pCtuUp && ((ctuRsAddr%frameWidthInCtus+1) < frameWidthInCtus)  )
      {
        TComDataCU *pCtuTR = pcPic->getCtu( ctuRsAddr - frameWidthInCtus + 1 );
        if ( pCtu->CUIsFromSameSliceAndTile(pCtuTR) )
        {
          // Top-right is available, so use it.
          pcSbacDecoder->loadContexts( &m_entropyCodingSyncContextState );
          for ( UChar comp = 0; comp < (pcSlice->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++ )
          {
            lastPaletteSize[comp] = m_entropyCodingSyncPaletteState.lastPaletteSize[comp];
            for ( UInt idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++ )
            {
              lastPalette[comp][idx] = m_entropyCodingSyncPaletteState.lastPalette[comp][idx];
            }
          }
        }
      }
    }

    for (UChar comp = 0; comp < (pcSlice->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++)
    {
      pCtu->setLastPaletteInLcuSizeFinal(comp, lastPaletteSize[comp]);
      for ( UInt idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++ )
      {
        pCtu->setLastPaletteInLcuFinal(comp, lastPalette[comp][idx], idx);
      }
    }

#if ENC_DEC_TRACE
    g_bJustDoIt = g_bEncDecTraceEnable;
#endif

#if DECODER_PARTIAL_CONFORMANCE_CHECK != 0
    const UInt numRemainingBitsPriorToCtu=ppcSubstreams[uiSubStrm]->getNumBitsLeft();
#endif

    if ( pcSlice->getSPS()->getUseSAO() )
    {
      SAOBlkParam& saoblkParam = (pcPic->getPicSym()->getSAOBlkParam())[ctuRsAddr];
      Bool bIsSAOSliceEnabled = false;
      Bool sliceEnabled[MAX_NUM_COMPONENT];
      for(Int comp=0; comp < MAX_NUM_COMPONENT; comp++)
      {
        ComponentID compId=ComponentID(comp);
        sliceEnabled[compId] = pcSlice->getSaoEnabledFlag(toChannelType(compId)) && (comp < pcPic->getNumberValidComponents());
        if (sliceEnabled[compId])
        {
          bIsSAOSliceEnabled=true;
        }
        saoblkParam[compId].modeIdc = SAO_MODE_OFF;
      }
      if (bIsSAOSliceEnabled)
      {
        Bool leftMergeAvail = false;
        Bool aboveMergeAvail= false;

        //merge left condition
        Int rx = (ctuRsAddr % frameWidthInCtus);
        if(rx > 0)
        {
          leftMergeAvail = pcPic->getSAOMergeAvailability(ctuRsAddr, ctuRsAddr-1);
        }
        //merge up condition
        Int ry = (ctuRsAddr / frameWidthInCtus);
        if(ry > 0)
        {
          aboveMergeAvail = pcPic->getSAOMergeAvailability(ctuRsAddr, ctuRsAddr-frameWidthInCtus);
        }

        pcSbacDecoder->parseSAOBlkParam( saoblkParam, sliceEnabled, leftMergeAvail, aboveMergeAvail, pcSlice->getSPS()->getBitDepths());
      }
    }

    m_pcCuDecoder->decodeCtu     ( pCtu, isLastCtuOfSliceSegment );

#if DECODER_PARTIAL_CONFORMANCE_CHECK != 0
    const UInt numRemainingBitsPostCtu=ppcSubstreams[uiSubStrm]->getNumBitsLeft(); // NOTE: Does not account for changes in buffered bits in CABAC decoder, although it's probably good enough.
    if (TDecConformanceCheck::doChecking() && m_pDecConformanceCheck)
    {
      m_pDecConformanceCheck->checkCtuDecoding(numRemainingBitsPriorToCtu-numRemainingBitsPostCtu);
    }
#endif

    m_pcCuDecoder->decompressCtu ( pCtu );

    if( pCtu->getLastPaletteInLcuSizeFinal( COMPONENT_Y ) )
    {
      for (UChar comp = 0; comp < (pCtu->getSlice()->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1 : 3); comp++)
      {
        lastPaletteSize[comp] = pCtu->getLastPaletteInLcuSizeFinal(comp);

        for (Int idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++)
        {
          lastPalette[comp][idx] = pCtu->getLastPaletteInLcuFinal(comp, idx);
        }
      }
    }

#if ENC_DEC_TRACE
    g_bJustDoIt = g_bEncDecTraceDisable;
#endif

    //Store probabilities of second CTU in line into buffer
    if ( ctuXPosInCtus == tileXPosInCtus+1 && wavefrontsEnabled)
    {
      m_entropyCodingSyncContextState.loadContexts( pcSbacDecoder );
      for ( UChar comp = 0; comp < (pcSlice->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1: 3); comp++ )
      {
        m_entropyCodingSyncPaletteState.lastPaletteSize[comp] = lastPaletteSize[comp];
        for ( UInt idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++ )
        {
          m_entropyCodingSyncPaletteState.lastPalette[comp][idx] = lastPalette[comp][idx];
        }
      }
    }

    if (isLastCtuOfSliceSegment)
    {
#if DECODER_CHECK_SUBSTREAM_AND_SLICE_TRAILING_BYTES
      pcSbacDecoder->parseRemainingBytes(false);
#endif
      if(!pcSlice->getDependentSliceSegmentFlag())
      {
        pcSlice->setSliceCurEndCtuTsAddr( ctuTsAddr+1 );
      }
      pcSlice->setSliceSegmentCurEndCtuTsAddr( ctuTsAddr+1 );
    }
    else if (  ctuXPosInCtus + 1 == tileXPosInCtus + currentTile.getTileWidthInCtus() &&
             ( ctuYPosInCtus + 1 == tileYPosInCtus + currentTile.getTileHeightInCtus() || wavefrontsEnabled)
            )
    {
      // The sub-stream/stream should be terminated after this CTU.
      // (end of slice-segment, end of tile, end of wavefront-CTU-row)
      UInt binVal;
      pcSbacDecoder->parseTerminatingBit( binVal );
      assert( binVal );
#if DECODER_CHECK_SUBSTREAM_AND_SLICE_TRAILING_BYTES
      pcSbacDecoder->parseRemainingBytes(true);
#endif
    }

  }

  assert(isLastCtuOfSliceSegment == true);


  if( depSliceSegmentsEnabled )
  {
    m_lastSliceSegmentEndContextState.loadContexts( pcSbacDecoder );//ctx end of dep.slice
    for ( UChar comp = 0; comp < (pcSlice->getSPS()->getChromaFormatIdc() == CHROMA_400 ? 1: 3); comp++ )
    {
      m_lastSliceSegmentEndPaletteState.lastPaletteSize[comp] = lastPaletteSize[comp];
      for ( UInt idx = 0; idx < pcSlice->getSPS()->getSpsScreenExtension().getPaletteMaxPredSize(); idx++ )
      {
        m_lastSliceSegmentEndPaletteState.lastPalette[comp][idx] = lastPalette[comp][idx];
      }
    }
  }

}


Void TDecSlice::xSetPredFromPPS(Pel lastPalette[MAX_NUM_COMPONENT][MAX_PALETTE_PRED_SIZE], UChar lastPaletteSize[MAX_NUM_COMPONENT], const TComPPS *pcPPS, const TComSPS *pcSPS)
{
  UInt num = std::min(pcPPS->getPpsScreenExtension().getNumPalettePred(), pcSPS->getSpsScreenExtension().getPaletteMaxPredSize());
  if (num == 0)
  {
    memset(lastPaletteSize, 0, MAX_NUM_COMPONENT*sizeof(UChar));
    return;
  }

  for(int i=0; i<(pcSPS->getChromaFormatIdc() == CHROMA_400 ? 1: 3); i++)
  {
    lastPaletteSize[i] = num;
    memcpy(lastPalette[i], pcPPS->getPpsScreenExtension().getPalettePred(i), num*sizeof(Pel));
  }
}

Void TDecSlice::xSetPredFromSPS(Pel lastPalette[MAX_NUM_COMPONENT][MAX_PALETTE_PRED_SIZE], UChar lastPaletteSize[MAX_NUM_COMPONENT], const TComSPS *pcSPS)
{
  UInt num = std::min(pcSPS->getSpsScreenExtension().getNumPalettePred(), pcSPS->getSpsScreenExtension().getPaletteMaxPredSize());
  if (num == 0)
  {
    memset(lastPaletteSize, 0, MAX_NUM_COMPONENT*sizeof(UChar));
    return;
  }

  for(int i=0; i<(pcSPS->getChromaFormatIdc() == CHROMA_400 ? 1: 3); i++)
  {
    lastPaletteSize[i] = num;
    memcpy(lastPalette[i], pcSPS->getSpsScreenExtension().getPalettePred(i), num*sizeof(Pel));
  }
}

Void TDecSlice::xSetPredDefault(Pel lastPalette[MAX_NUM_COMPONENT][MAX_PALETTE_PRED_SIZE], UChar lastPaletteSize[MAX_NUM_COMPONENT], const TComSPS *pcSPS)
{
  for(int i=0; i<(pcSPS->getChromaFormatIdc() == CHROMA_400 ? 1: 3); i++)
  {
    lastPaletteSize[i] = 0;
    memset(lastPalette[i],0, pcSPS->getSpsScreenExtension().getPaletteMaxSize()*sizeof(Pel));
  }
}

//! \}
