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

/** \file     TComHash.h
    \brief    common Hash class (header)
*/

#ifndef __TCOMHASH__
#define __TCOMHASH__

// Include files
#include "CommonDef.h"
#include "TComPicSym.h"
#include "TComPicYuv.h"
#include <vector>

//! \ingroup TLibEncoder
//! \{

struct BlockHash
{
  Short x;
  Short y;
  UInt hashValue2;
};

typedef vector<BlockHash>::iterator MapIterator;

// ====================================================================================================================
// Class definitions
// ====================================================================================================================


class TCRCCalculatorLight
{
public:
  TCRCCalculatorLight( UInt bits, UInt truncPoly );
  ~TCRCCalculatorLight();

public:
  Void processData( UChar* pData, UInt dataLength );
  Void reset() { m_remainder = 0; }
  UInt getCRC() { return m_remainder & m_finalResultMask; }

private:
  Void xInitTable();

private:
  UInt m_remainder;
  UInt m_truncPoly;
  UInt m_bits;
  UInt m_table[256];
  UInt m_finalResultMask;
};


class TComHash
{
public:
  TComHash();
  ~TComHash();
  Void create();
  Void clearAll();
  Void addToTable( UInt hashValue, const BlockHash& blockHash );
  Int count( UInt hashValue );
  Int count( UInt hashValue ) const;
  MapIterator getFirstIterator( UInt hashValue );
  const MapIterator getFirstIterator( UInt hashValue ) const;
  Bool hasExactMatch( UInt hashValue1, UInt hashValue2 );

  Void generateBlock2x2HashValue( TComPicYuv* pPicYuv, Int picWidth, Int picHeight, const BitDepths bitDepths, UInt* picBlockHash[2], Bool* picBlockSameInfo[3]);
  Void generateBlockHashValue( Int picWidth, Int picHeight, Int width, Int height, UInt* srcPicBlockHash[2], UInt* dstPicBlockHash[2], Bool* srcPicBlockSameInfo[3], Bool* dstPicBlockSameInfo[3]);
  Void addToHashMapByRowWithPrecalData( UInt* srcHash[2], Bool* srcIsSame, Int picWidth, Int picHeight, Int width, Int height);

public:
  static UInt   getCRCValue1( UChar* p, Int length );
  static UInt   getCRCValue2( UChar* p, Int length );
  static Void getPixelsIn1DCharArrayByBlock2x2(const TComPicYuv* const pPicYuv, UChar* pPixelsIn1D, Int xStart, Int yStart, const BitDepths& bitDepths, Bool includeAllComponent = true);
  static Bool isBlock2x2RowSameValue(UChar* p, Bool includeAllComponent = true);
  static Bool isBlock2x2ColSameValue(UChar* p, Bool includeAllComponent = true);
  static Bool isHorizontalPerfect( TComPicYuv* pPicYuv, Int width, Int height, Int xStart, Int yStart, Bool includeAllComponent = true );
  static Bool isVerticalPerfect  ( TComPicYuv* pPicYuv, Int width, Int height, Int xStart, Int yStart, Bool includeAllComponent = true );
  static Bool getBlockHashValue( const TComPicYuv* const pPicYuv, Int width, Int height, Int xStart, Int yStart, const BitDepths bitDepths, UInt& hashValue1, UInt& hashValue2 );
  static Void initBlockSizeToIndex();

private:
  vector<BlockHash>** m_pLookupTable;

private:
  static const Int m_CRCBits = 16;
  static const Int m_blockSizeBits = 2;
  static Int m_blockSizeToIndex[65][65];

  static TCRCCalculatorLight m_crcCalculator1;
  static TCRCCalculatorLight m_crcCalculator2;
};

//! \}

#endif // __TCOMHASH__
