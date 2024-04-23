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

/** \file     TEncHash.cpp
    \brief    hash encoder class
*/

#include "CommonDef.h"
#include "TComHash.h"

//! \ingroup TLibCommon
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

Int TComHash::m_blockSizeToIndex[65][65];
TCRCCalculatorLight TComHash::m_crcCalculator1(24, 0x5D6DCB);
TCRCCalculatorLight TComHash::m_crcCalculator2(24, 0x864CFB);

TCRCCalculatorLight::TCRCCalculatorLight( UInt bits, UInt truncPoly )
{
  m_remainder = 0;
  m_bits = bits;
  m_truncPoly = truncPoly;
  m_finalResultMask = (1<<bits) - 1;

  xInitTable();
}

TCRCCalculatorLight::~TCRCCalculatorLight()
{

}

Void TCRCCalculatorLight::xInitTable()
{
  const UInt highBit = 1<<(m_bits-1);
  const UInt ByteHighBit  = 1<<(8-1);

  for ( UInt value = 0; value < 256; value++ )
  {
    UInt remainder = 0;
    for ( UChar mask = ByteHighBit; mask != 0; mask >>= 1 )
    {
      if ( value & mask )
      {
        remainder ^= highBit;
      }

      if ( remainder & highBit )
      {
        remainder <<= 1;
        remainder ^= m_truncPoly;
      }
      else
      {
        remainder <<= 1;
      }
    }

    m_table[value] = remainder;
  }
}

Void TCRCCalculatorLight::processData( UChar* pData, UInt dataLength )
{
  for ( UInt i=0; i<dataLength; i++ )
  {
    UChar index = ( m_remainder>>(m_bits-8) ) ^ pData[i];
    m_remainder <<= 8;
    m_remainder ^= m_table[index];
  }
}


TComHash::TComHash()
{
  m_pLookupTable = NULL;
}

TComHash::~TComHash()
{
  clearAll();
  if ( m_pLookupTable != NULL )
  {
    delete[] m_pLookupTable;
  }
}

Void TComHash::create()
{
  if ( m_pLookupTable != NULL )
  {
    clearAll();
    return;
  }
  Int maxAddr = 1<<(m_CRCBits+m_blockSizeBits);
  m_pLookupTable = new vector<BlockHash>*[maxAddr];
  memset( m_pLookupTable, 0, sizeof(vector<BlockHash>*) * maxAddr );
}

Void TComHash::clearAll()
{
  if ( m_pLookupTable == NULL )
  {
    return;
  }
  Int maxAddr = 1<<(m_CRCBits+m_blockSizeBits);
  for ( Int i=0; i<maxAddr; i++ )
  {
    if ( m_pLookupTable[i] != NULL )
    {
      delete m_pLookupTable[i];
      m_pLookupTable[i] = NULL;
    }
  }
}

Void TComHash::addToTable( UInt hashValue, const BlockHash& blockHash )
{
  if ( m_pLookupTable[hashValue] == NULL )
  {
    m_pLookupTable[hashValue] = new vector<BlockHash>;
    m_pLookupTable[hashValue]->push_back( blockHash );
  }
  else
  {
    m_pLookupTable[hashValue]->push_back( blockHash );
  }
}

Int TComHash::count( UInt hashValue )
{
  if ( m_pLookupTable[hashValue] == NULL )
  {
    return 0;
  }
  else
  {
    return static_cast<Int>( m_pLookupTable[hashValue]->size() );
  }
}

Int TComHash::count( UInt hashValue ) const
{
  if ( m_pLookupTable[hashValue] == NULL )
  {
    return 0;
  }
  else
  {
    return static_cast<Int>( m_pLookupTable[hashValue]->size() );
  }
}

MapIterator TComHash::getFirstIterator( UInt hashValue )
{
  assert( count( hashValue ) > 0 );
  return m_pLookupTable[hashValue]->begin();
}

const MapIterator TComHash::getFirstIterator( UInt hashValue ) const
{
  assert( count( hashValue ) > 0 );
  return m_pLookupTable[hashValue]->begin();
}

Bool TComHash::hasExactMatch( UInt hashValue1, UInt hashValue2 )
{
  if ( m_pLookupTable[hashValue1] == NULL )
  {
    return false;
  }
  vector<BlockHash>::iterator it;
  for ( it = m_pLookupTable[hashValue1]->begin(); it != m_pLookupTable[hashValue1]->end(); it++ )
  {
    if ( (*it).hashValue2 == hashValue2 )
    {
      return true;
    }
  }
  return false;
}

Void TComHash::generateBlock2x2HashValue( TComPicYuv* pPicYuv, Int picWidth, Int picHeight, const BitDepths bitDepths, UInt* picBlockHash[2], Bool* picBlockSameInfo[3])
{
  const Int width = 2;
  const Int height = 2;
  Int xEnd = picWidth - width + 1;
  Int yEnd = picHeight - height + 1;

  Int length = width * 2;
  Bool bIncludeChroma = false;
  if (pPicYuv->getChromaFormat() == CHROMA_444)
  {
    length *= 3;
    bIncludeChroma = true;
  }
  UChar* p = new UChar[length];

  Int pos = 0;
  for (Int yPos = 0; yPos < yEnd; yPos++)
  {
    for (Int xPos = 0; xPos < xEnd; xPos++)
    {
      TComHash::getPixelsIn1DCharArrayByBlock2x2(pPicYuv, p, xPos, yPos, bitDepths, bIncludeChroma);
      picBlockSameInfo[0][pos] = isBlock2x2RowSameValue(p, bIncludeChroma);
      picBlockSameInfo[1][pos] = isBlock2x2ColSameValue(p, bIncludeChroma);

      picBlockHash[0][pos] = TComHash::getCRCValue1(p, length * sizeof(UChar));
      picBlockHash[1][pos] = TComHash::getCRCValue2(p, length * sizeof(UChar));

      pos++;
    }
    pos += width - 1;
  }

  delete[] p;
}

Void TComHash::generateBlockHashValue( Int picWidth, Int picHeight, Int width, Int height, UInt* srcPicBlockHash[2], UInt* dstPicBlockHash[2], Bool* srcPicBlockSameInfo[3], Bool* dstPicBlockSameInfo[3])
{
  Int xEnd = picWidth - width + 1;
  Int yEnd = picHeight - height + 1;

  Int srcWidth = width >> 1;
  Int quadWidth = width >> 2;
  Int srcHeight = height >> 1;
  Int quadHeight = height >> 2;

  Int length = 4 * sizeof(UInt);

  UInt* p = new UInt[4];
  Int pos = 0;
  for (Int yPos = 0; yPos < yEnd; yPos++)
  {
    for (Int xPos = 0; xPos < xEnd; xPos++)
    {
      p[0] = srcPicBlockHash[0][pos];
      p[1] = srcPicBlockHash[0][pos + srcWidth];
      p[2] = srcPicBlockHash[0][pos + srcHeight*picWidth];
      p[3] = srcPicBlockHash[0][pos + srcHeight*picWidth + srcWidth];
      dstPicBlockHash[0][pos] = TComHash::getCRCValue1((UChar*)p, length);

      p[0] = srcPicBlockHash[1][pos];
      p[1] = srcPicBlockHash[1][pos + srcWidth];
      p[2] = srcPicBlockHash[1][pos + srcHeight*picWidth];
      p[3] = srcPicBlockHash[1][pos + srcHeight*picWidth + srcWidth];
      dstPicBlockHash[1][pos] = TComHash::getCRCValue2((UChar*)p, length);

      dstPicBlockSameInfo[0][pos] = srcPicBlockSameInfo[0][pos] && srcPicBlockSameInfo[0][pos + quadWidth] && srcPicBlockSameInfo[0][pos + srcWidth]
        && srcPicBlockSameInfo[0][pos + srcHeight*picWidth] && srcPicBlockSameInfo[0][pos + srcHeight*picWidth + quadWidth] && srcPicBlockSameInfo[0][pos + srcHeight*picWidth + srcWidth];

      dstPicBlockSameInfo[1][pos] = srcPicBlockSameInfo[1][pos] && srcPicBlockSameInfo[1][pos + srcWidth] && srcPicBlockSameInfo[1][pos + quadHeight*picWidth]
        && srcPicBlockSameInfo[1][pos + quadHeight*picWidth + srcWidth] && srcPicBlockSameInfo[1][pos + srcHeight*picWidth] && srcPicBlockSameInfo[1][pos + srcHeight*picWidth + srcWidth];

      pos++;
    }
    pos += width - 1;
  }

  if (width >= 8)
  {
    Int widthMinus1 = width - 1;
    Int heightMinus1 = height - 1;
    pos = 0;

    for (Int yPos = 0; yPos < yEnd; yPos++)
    {
      for (Int xPos = 0; xPos < xEnd; xPos++)
      {
        dstPicBlockSameInfo[2][pos] = (!dstPicBlockSameInfo[0][pos] && !dstPicBlockSameInfo[1][pos]) || (((xPos & widthMinus1) == 0) && ((yPos & heightMinus1) == 0));
        pos++;
      }
      pos += width - 1;
    }
  }

  delete[] p;

}

Void TComHash::addToHashMapByRowWithPrecalData( UInt* picHash[2], Bool* picIsSame, Int picWidth, Int picHeight, Int width, Int height)
{
  Int xEnd = picWidth - width + 1;
  Int yEnd = picHeight - height + 1;

  Bool* srcIsAdded = picIsSame;
  UInt* srcHash[2] = { picHash[0], picHash[1] };

  Int addValue = m_blockSizeToIndex[width][height];
  assert(addValue >= 0);
  addValue <<= m_CRCBits;
  Int crcMask = 1 << m_CRCBits;
  crcMask -= 1;

  for (Int xPos = 0; xPos < xEnd; xPos++)
  {
    for (Int yPos = 0; yPos < yEnd; yPos++)
    {
      Int pos = yPos*picWidth + xPos;
      //valid data
      if (srcIsAdded[pos])
      {
        BlockHash blockHash;
        blockHash.x = xPos;
        blockHash.y = yPos;

        UInt      hashValue1 = (srcHash[0][pos] & crcMask) + addValue;
        blockHash.hashValue2 = srcHash[1][pos];

        addToTable(hashValue1, blockHash);
      }
    }
  }
}

Void TComHash::getPixelsIn1DCharArrayByBlock2x2( const TComPicYuv* const pPicYuv, UChar* pPixelsIn1D, Int xStart, Int yStart, const BitDepths& bitDepths, Bool includeAllComponent)
{
  if ( pPicYuv->getChromaFormat() != CHROMA_444 )
  {
    includeAllComponent = false;
  }

  if ( bitDepths.recon[CHANNEL_TYPE_LUMA] == 8 && bitDepths.recon[CHANNEL_TYPE_CHROMA] == 8 )
  {
    Pel* pPel[3];
    Int stride[3];
    for ( Int id=0; id<3; id++ )
    {
      ComponentID compID = ComponentID( id );
      stride[id] = pPicYuv->getStride( compID );
      pPel[id] = const_cast<Pel*>( pPicYuv->getAddr( compID ) );
      pPel[id] += (yStart >> pPicYuv->getComponentScaleX( compID )) * stride[id] + (xStart >> pPicYuv->getComponentScaleX( compID ));
    }

    Int index = 0;
    for (Int i = 0; i < 2; i++)
    {
      for (Int j = 0; j < 2; j++)
      {
        pPixelsIn1D[index++] = static_cast<UChar>(pPel[0][j]);
        if (includeAllComponent)
        {
          pPixelsIn1D[index++] = static_cast<UChar>(pPel[1][j]);
          pPixelsIn1D[index++] = static_cast<UChar>(pPel[2][j]);
        }
      }
      pPel[0] += stride[0];
      if (includeAllComponent)
      {
        pPel[1] += stride[1];
        pPel[2] += stride[2];
      }
    }
  }
  else
  {
    Int shift = bitDepths.recon[CHANNEL_TYPE_LUMA] - 8;
    Int shiftc = bitDepths.recon[CHANNEL_TYPE_CHROMA] - 8;
    Pel* pPel[3];
    Int stride[3];
    for ( Int id=0; id<3; id++ )
    {
      ComponentID compID = ComponentID( id );
      stride[id] = pPicYuv->getStride( compID );
      pPel[id] = const_cast<Pel*>( pPicYuv->getAddr( compID ) );
      pPel[id] += (yStart >> pPicYuv->getComponentScaleX( compID )) * stride[id] + (xStart >> pPicYuv->getComponentScaleX( compID ));
    }

    Int index = 0;
    for (Int i = 0; i < 2; i++)
    {
      for (Int j = 0; j < 2; j++)
      {
        pPixelsIn1D[index++] = static_cast<UChar>(pPel[0][j] >> shift);
        if (includeAllComponent)
        {
          pPixelsIn1D[index++] = static_cast<UChar>(pPel[1][j] >> shiftc);
          pPixelsIn1D[index++] = static_cast<UChar>(pPel[2][j] >> shiftc);
        }
      }
      pPel[0] += stride[0];
      if (includeAllComponent)
      {
        pPel[1] += stride[1];
        pPel[2] += stride[2];
      }
    }
  }
}

Bool TComHash::isBlock2x2RowSameValue( UChar* p, Bool includeAllComponent )
{
  if ( includeAllComponent )
  {
    if ( p[0] != p[3] || p[6] != p[9] )
    {
      return false;
    }
    if ( p[1] != p[4] || p[7] != p[10] )
    {
      return false;
    }
    if ( p[2] != p[5] || p[8] != p[11] )
    {
      return false;
    }
  }
  else
  {
    if ( p[0] != p[1] || p[2] != p[3] )
    {
      return false;
    }
  }

  return true;
}

Bool TComHash::isBlock2x2ColSameValue( UChar* p, Bool includeAllComponent )
{
  if ( includeAllComponent )
  {
    if ( p[0] != p[6] || p[3] != p[9] )
    {
      return false;
    }
    if ( p[1] != p[7] || p[4] != p[10] )
    {
      return false;
    }
    if ( p[2] != p[8] || p[5] != p[11] )
    {
      return false;
    }
  }
  else
  {
    if ( (p[0] != p[2]) || (p[1] != p[3]) )
    {
      return false;
    }
  }

  return true;
}

Bool TComHash::isHorizontalPerfect( TComPicYuv* pPicYuv, Int width, Int height, Int xStart, Int yStart, Bool includeAllComponent )
{
  if ( pPicYuv->getChromaFormat() != CHROMA_444 )
  {
    includeAllComponent = false;
  }

  for ( Int id=0; id < (includeAllComponent ? 3 : 1); id++ )
  {
    ComponentID compID = ComponentID( id );
    Int stride = pPicYuv->getStride( compID );
    Pel* p = pPicYuv->getAddr( compID );
    p += (yStart >> pPicYuv->getComponentScaleX( compID )) * stride + (xStart >> pPicYuv->getComponentScaleX( compID ));

    for ( Int i=0; i<height; i++ )
    {
      for ( Int j=1; j<width; j++ )
      {
        if ( p[j] != p[0] )
        {
          return false;
        }
      }
      p += stride;
    }
  }

  return true;
}

Bool TComHash::isVerticalPerfect( TComPicYuv* pPicYuv, Int width, Int height, Int xStart, Int yStart, Bool includeAllComponent )
{
  if ( pPicYuv->getChromaFormat() != CHROMA_444 )
  {
    includeAllComponent = false;
  }

  for ( Int id=0; id < (includeAllComponent ? 3 : 1); id++ )
  {
    ComponentID compID = ComponentID( id );
    Int stride = pPicYuv->getStride( compID );
    Pel* p = pPicYuv->getAddr( compID );
    p += (yStart >> pPicYuv->getComponentScaleX( compID )) * stride + (xStart >> pPicYuv->getComponentScaleX( compID ));

    for ( Int i=0; i<width; i++ )
    {
      for ( Int j=1; j<height; j++ )
      {
        if ( p[j*stride+i] != p[i] )
        {
          return false;
        }
      }
    }
  }

  return true;
}

Bool TComHash::getBlockHashValue( const TComPicYuv* const pPicYuv, Int width, Int height, Int xStart, Int yStart, const BitDepths bitDepths, UInt& hashValue1, UInt& hashValue2 )
{
  Int addValue = m_blockSizeToIndex[width][height];
  assert( addValue >= 0 );
  addValue <<= m_CRCBits;
  Int crcMask = 1<<m_CRCBits;
  crcMask -= 1;
  Int length = 4;
  Bool bIncludeChroma = false;
  if (pPicYuv->getChromaFormat() == CHROMA_444)
  {
    length *= 3;
    bIncludeChroma = true;
  }

  UChar* p = new UChar[length];
  UInt* toHash = new UInt[4];

  Int block2x2Num = (width*height) >> 2;

  UInt* hashValueBuffer[2][2];
  for (Int i = 0; i < 2; i++)
  {
    for (Int j = 0; j < 2; j++)
    {
      hashValueBuffer[i][j] = new UInt[block2x2Num];
    }
  }

  //2x2 subblock hash values in current CU
  Int subBlockInWidth = (width >> 1);
  for (Int yPos = 0; yPos < height; yPos += 2)
  {
    for (Int xPos = 0; xPos < width; xPos += 2)
    {
      Int pos = (yPos >> 1)*subBlockInWidth + (xPos >> 1);
      TComHash::getPixelsIn1DCharArrayByBlock2x2(pPicYuv, p, xStart + xPos, yStart + yPos, bitDepths, bIncludeChroma);

      hashValueBuffer[0][0][pos] = TComHash::getCRCValue1(p, length * sizeof(UChar));
      hashValueBuffer[1][0][pos] = TComHash::getCRCValue2(p, length * sizeof(UChar));
    }
  }

  Int srcSubBlockInWidth = subBlockInWidth;
  subBlockInWidth >>= 1;
  length = 4 * sizeof(UInt);

  Int srcIdx = 1;
  Int dstIdx = 0;

  //4x4 subblock hash values to current block hash values
  for (Int subWidth = 4; subWidth <= width; subWidth *= 2)
  {
    srcIdx = 1 - srcIdx;
    dstIdx = 1 - dstIdx;

    Int dstPos = 0;
    for (Int yPos = 0; yPos < subBlockInWidth; yPos++)
    {
      for (Int xPos = 0; xPos < subBlockInWidth; xPos++)
      {
        Int srcPos = (yPos << 1)*srcSubBlockInWidth + (xPos << 1);

        toHash[0] = hashValueBuffer[0][srcIdx][srcPos];
        toHash[1] = hashValueBuffer[0][srcIdx][srcPos + 1];
        toHash[2] = hashValueBuffer[0][srcIdx][srcPos + srcSubBlockInWidth];
        toHash[3] = hashValueBuffer[0][srcIdx][srcPos + srcSubBlockInWidth + 1];

        hashValueBuffer[0][dstIdx][dstPos] = TComHash::getCRCValue1((UChar*)toHash, length);

        toHash[0] = hashValueBuffer[1][srcIdx][srcPos];
        toHash[1] = hashValueBuffer[1][srcIdx][srcPos + 1];
        toHash[2] = hashValueBuffer[1][srcIdx][srcPos + srcSubBlockInWidth];
        toHash[3] = hashValueBuffer[1][srcIdx][srcPos + srcSubBlockInWidth + 1];
        hashValueBuffer[1][dstIdx][dstPos] = TComHash::getCRCValue2((UChar*)toHash, length);

        dstPos++;
      }
    }

    srcSubBlockInWidth = subBlockInWidth;
    subBlockInWidth >>= 1;
  }

  hashValue1 = (hashValueBuffer[0][dstIdx][0] & crcMask) + addValue;
  hashValue2 = hashValueBuffer[1][dstIdx][0];

  delete[] toHash;

  for (Int i = 0; i < 2; i++)
  {
    for (Int j = 0; j < 2; j++)
    {
      delete[] hashValueBuffer[i][j];
    }
  }

  delete[] p;

  return true;
}

Void TComHash::initBlockSizeToIndex()
{
  for ( Int i=0; i<65; i++ )
  {
    for ( Int j=0; j<65; j++ )
    {
      m_blockSizeToIndex[i][j] = -1;
    }
  }

  m_blockSizeToIndex[ 8][ 8] = 0;
  m_blockSizeToIndex[16][16] = 1;
  m_blockSizeToIndex[32][32] = 2;
  m_blockSizeToIndex[64][64] = 3;
}

UInt TComHash::getCRCValue1( UChar* p, Int length )
{
  m_crcCalculator1.reset();
  m_crcCalculator1.processData( p, length );
  return m_crcCalculator1.getCRC();
}

UInt TComHash::getCRCValue2( UChar* p, Int length )
{
  m_crcCalculator2.reset();
  m_crcCalculator2.processData( p, length );
  return m_crcCalculator2.getCRC();
}

//! \}
