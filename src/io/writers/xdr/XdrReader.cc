
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "io/writers/xdr/XdrReader.h"
#include "Exception.h"
#include <cstdlib>
#include <iostream>

namespace hemelb
{
  namespace io
  {
    namespace writers
    {
      namespace xdr
      {

        XdrReader::XdrReader()
        {
        }

        // Functions to read out the next bit of the file as a certain type.
        bool XdrReader::readDouble(double& outDouble)
        {
          return xdr_double(&mXdr, &outDouble);
        }

        bool XdrReader::readFloat(float& outDouble)
        {
          return xdr_float(&mXdr, &outDouble);
        }

        bool XdrReader::readInt(int& outInt)
        {
          return xdr_int(&mXdr, &outInt);
        }

        bool XdrReader::readUnsignedInt(unsigned int& outUInt)
        {
          return xdr_u_int(&mXdr, &outUInt);
        }

        bool XdrReader::readUnsignedLong(uint64_t& outULong)
        {
          u_quad_t temporary;
          bool ret = xdr_uint64_t(&mXdr, &temporary);
          outULong = temporary;
          return ret;
        }

        unsigned int XdrReader::GetPosition()
        {
          return xdr_getpos(&mXdr);
        }

        // Returns false on failure
        bool XdrReader::SetPosition(unsigned int iPosition)
        {
          return xdr_setpos(&mXdr, iPosition);
        }

        // Destructor to get rid of any resources used by the Xdr object. This class doesn't create
        // the file object, so it doesn't free it either.
        XdrReader::~XdrReader()
        {
          xdr_destroy(&mXdr);
        }

	// Specialisations to delegate to the XDR API
	template<>
	bool XdrReader::read<double>(double& val) {
	  return xdr_double(&mXdr, &val);
	}
	template<>
	bool XdrReader::read<float>(float& val) {
	  return xdr_float(&mXdr, &val);
	}
	template<>
	bool XdrReader::read<int>(int& val) {
	  return xdr_int(&mXdr, &val);
	}
	template<>
	bool XdrReader::read<unsigned int>(unsigned int& val) {
	  return xdr_u_int(&mXdr, &val);
	}
	template<>
	bool XdrReader::read<uint64_t>(uint64_t& val) {
	  static_assert(std::is_same<uint64_t, u_quad_t>::value, "uint64_t");
	  return xdr_uint64_t(&mXdr, &val);
	}
	template<>
	bool XdrReader::read<std::string>(std::string& val) {
	  char* ans = NULL;
	  bool ok = xdr_wrapstring(&mXdr, &ans);
	  if (ok)
	    val = ans;

	  if (ans)
	    std::free(ans);

	  return ok;
        }

      } // namespace xdr
    } // namespace writers
  }
}
