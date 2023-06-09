// Copyright (c) 1995-2009 held by the author(s).  All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer
//   in the documentation and/or other materials provided with the
//   distribution.
// * Neither the names of the Naval Postgraduate School (NPS)
//   Modeling Virtual Environments and Simulation (MOVES) Institute
//   (http://www.nps.edu and http://www.MovesInstitute.org)
//   nor the names of its contributors may be used to endorse or
//   promote products derived from this software without specific
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008, MOVES Institute, Naval Postgraduate School. All 
// rights reserved. This work is licensed under the BSD open source license,
// available at https://www.movesinstitute.org/licenses/bsd.html
//
// Author: DMcG
// Modified for use with C#:
//  - Peter Smith (Naval Air Warfare Center - Training Systems Division)
//  - Zvonko Bostjancic (Blubit d.o.o. - zvonko.bostjancic@blubit.si)

using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using System.Xml.Serialization;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using OpenDis.Core;

namespace OpenDis.Dis2012
{
    /// <summary>
    ///  Information about a geometry, a state associated with a geometry, a bounding volume, or an associated entity ID. NOTE: this class requires hand coding. 6.2.31
    /// </summary>
    [Serializable]
    [XmlRoot]
    public partial class EnvironmentGeneral
    {
        /// <summary>
        /// Record type
        /// </summary>
        private uint _environmentType;

        /// <summary>
        /// length, in bits
        /// </summary>
        private byte _length;

        /// <summary>
        /// Identify the sequentially numbered record index
        /// </summary>
        private byte _index;

        /// <summary>
        /// padding
        /// </summary>
        private byte _padding1;

        /// <summary>
        /// Geometry or state record
        /// </summary>
        private byte _geometry;

        /// <summary>
        /// padding to bring the total size up to a 64 bit boundry
        /// </summary>
        private byte _padding2;

        /// <summary>
        /// Initializes a new instance of the <see cref="EnvironmentGeneral"/> class.
        /// </summary>
        public EnvironmentGeneral()
        {
        }

        /// <summary>
        /// Implements the operator !=.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        /// 	<c>true</c> if operands are not equal; otherwise, <c>false</c>.
        /// </returns>
        public static bool operator !=(EnvironmentGeneral left, EnvironmentGeneral right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Implements the operator ==.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public static bool operator ==(EnvironmentGeneral left, EnvironmentGeneral right)
        {
            if (object.ReferenceEquals(left, right))
            {
                return true;
            }

            if (((object)left == null) || ((object)right == null))
            {
                return false;
            }

            return left.Equals(right);
        }

        public virtual int GetMarshalledSize()
        {
            int marshalSize = 0; 

            marshalSize += 4;  // this._environmentType
            marshalSize += 1;  // this._length
            marshalSize += 1;  // this._index
            marshalSize += 1;  // this._padding1
            marshalSize += 1;  // this._geometry
            marshalSize += 1;  // this._padding2
            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the Record type
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "environmentType")]
        public uint EnvironmentType
        {
            get
            {
                return this._environmentType;
            }

            set
            {
                this._environmentType = value;
            }
        }

        /// <summary>
        /// Gets or sets the length, in bits
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "length")]
        public byte Length
        {
            get
            {
                return this._length;
            }

            set
            {
                this._length = value;
            }
        }

        /// <summary>
        /// Gets or sets the Identify the sequentially numbered record index
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "index")]
        public byte Index
        {
            get
            {
                return this._index;
            }

            set
            {
                this._index = value;
            }
        }

        /// <summary>
        /// Gets or sets the padding
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "padding1")]
        public byte Padding1
        {
            get
            {
                return this._padding1;
            }

            set
            {
                this._padding1 = value;
            }
        }

        /// <summary>
        /// Gets or sets the Geometry or state record
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "geometry")]
        public byte Geometry
        {
            get
            {
                return this._geometry;
            }

            set
            {
                this._geometry = value;
            }
        }

        /// <summary>
        /// Gets or sets the padding to bring the total size up to a 64 bit boundry
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "padding2")]
        public byte Padding2
        {
            get
            {
                return this._padding2;
            }

            set
            {
                this._padding2 = value;
            }
        }

        /// <summary>
        /// Occurs when exception when processing PDU is caught.
        /// </summary>
        public event Action<Exception> Exception;

        /// <summary>
        /// Called when exception occurs (raises the <see cref="Exception"/> event).
        /// </summary>
        /// <param name="e">The exception.</param>
        protected void OnException(Exception e)
        {
            if (this.Exception != null)
            {
                this.Exception(e);
            }
        }

        /// <summary>
        /// Marshal the data to the DataOutputStream.  Note: Length needs to be set before calling this method
        /// </summary>
        /// <param name="dos">The DataOutputStream instance to which the PDU is marshaled.</param>
        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes", Justification = "Due to ignoring errors.")]
        public virtual void Marshal(DataOutputStream dos)
        {
            if (dos != null)
            {
                try
                {
                    dos.WriteUnsignedInt((uint)this._environmentType);
                    dos.WriteUnsignedByte((byte)this._length);
                    dos.WriteUnsignedByte((byte)this._index);
                    dos.WriteUnsignedByte((byte)this._padding1);
                    dos.WriteUnsignedByte((byte)this._geometry);
                    dos.WriteUnsignedByte((byte)this._padding2);
                }
                catch (Exception e)
                {
#if DEBUG
                    Trace.WriteLine(e);
                    Trace.Flush();
#endif
                    this.OnException(e);
                }
            }
        }

        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes", Justification = "Due to ignoring errors.")]
        public virtual void Unmarshal(DataInputStream dis)
        {
            if (dis != null)
            {
                try
                {
                    this._environmentType = dis.ReadUnsignedInt();
                    this._length = dis.ReadUnsignedByte();
                    this._index = dis.ReadUnsignedByte();
                    this._padding1 = dis.ReadUnsignedByte();
                    this._geometry = dis.ReadUnsignedByte();
                    this._padding2 = dis.ReadUnsignedByte();
                }
                catch (Exception e)
                {
#if DEBUG
                    Trace.WriteLine(e);
                    Trace.Flush();
#endif
                    this.OnException(e);
                }
            }
        }

        /// <summary>
        /// This allows for a quick display of PDU data.  The current format is unacceptable and only used for debugging.
        /// This will be modified in the future to provide a better display.  Usage: 
        /// pdu.GetType().InvokeMember("Reflection", System.Reflection.BindingFlags.InvokeMethod, null, pdu, new object[] { sb });
        /// where pdu is an object representing a single pdu and sb is a StringBuilder.
        /// Note: The supplied Utilities folder contains a method called 'DecodePDU' in the PDUProcessor Class that provides this functionality
        /// </summary>
        /// <param name="sb">The StringBuilder instance to which the PDU is written to.</param>
        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes", Justification = "Due to ignoring errors.")]
        public virtual void Reflection(StringBuilder sb)
        {
            sb.AppendLine("<EnvironmentGeneral>");
            try
            {
                sb.AppendLine("<environmentType type=\"uint\">" + this._environmentType.ToString(CultureInfo.InvariantCulture) + "</environmentType>");
                sb.AppendLine("<length type=\"byte\">" + this._length.ToString(CultureInfo.InvariantCulture) + "</length>");
                sb.AppendLine("<index type=\"byte\">" + this._index.ToString(CultureInfo.InvariantCulture) + "</index>");
                sb.AppendLine("<padding1 type=\"byte\">" + this._padding1.ToString(CultureInfo.InvariantCulture) + "</padding1>");
                sb.AppendLine("<geometry type=\"byte\">" + this._geometry.ToString(CultureInfo.InvariantCulture) + "</geometry>");
                sb.AppendLine("<padding2 type=\"byte\">" + this._padding2.ToString(CultureInfo.InvariantCulture) + "</padding2>");
                sb.AppendLine("</EnvironmentGeneral>");
            }
            catch (Exception e)
            {
#if DEBUG
                    Trace.WriteLine(e);
                    Trace.Flush();
#endif
                    this.OnException(e);
            }
        }

        /// <summary>
        /// Determines whether the specified <see cref="System.Object"/> is equal to this instance.
        /// </summary>
        /// <param name="obj">The <see cref="System.Object"/> to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if the specified <see cref="System.Object"/> is equal to this instance; otherwise, <c>false</c>.
        /// </returns>
        public override bool Equals(object obj)
        {
            return this == obj as EnvironmentGeneral;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(EnvironmentGeneral obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            if (this._environmentType != obj._environmentType)
            {
                ivarsEqual = false;
            }

            if (this._length != obj._length)
            {
                ivarsEqual = false;
            }

            if (this._index != obj._index)
            {
                ivarsEqual = false;
            }

            if (this._padding1 != obj._padding1)
            {
                ivarsEqual = false;
            }

            if (this._geometry != obj._geometry)
            {
                ivarsEqual = false;
            }

            if (this._padding2 != obj._padding2)
            {
                ivarsEqual = false;
            }

            return ivarsEqual;
        }

        /// <summary>
        /// HashCode Helper
        /// </summary>
        /// <param name="hash">The hash value.</param>
        /// <returns>The new hash value.</returns>
        private static int GenerateHash(int hash)
        {
            hash = hash << (5 + hash);
            return hash;
        }

        /// <summary>
        /// Gets the hash code.
        /// </summary>
        /// <returns>The hash code.</returns>
        public override int GetHashCode()
        {
            int result = 0;

            result = GenerateHash(result) ^ this._environmentType.GetHashCode();
            result = GenerateHash(result) ^ this._length.GetHashCode();
            result = GenerateHash(result) ^ this._index.GetHashCode();
            result = GenerateHash(result) ^ this._padding1.GetHashCode();
            result = GenerateHash(result) ^ this._geometry.GetHashCode();
            result = GenerateHash(result) ^ this._padding2.GetHashCode();

            return result;
        }
    }
}
