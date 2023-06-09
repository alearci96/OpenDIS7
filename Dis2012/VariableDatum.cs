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
    /// the variable datum type, the datum length, and the value for that variable datum type. NOT COMPLETE. Section 6.2.92
    /// </summary>
    [Serializable]
    [XmlRoot]
    public partial class VariableDatum
    {
        /// <summary>
        /// Type of variable datum to be transmitted. 32 bit enumeration defined in EBV
        /// </summary>
        private uint _variableDatumID;

        /// <summary>
        /// Length, in bits, of the variable datum.
        /// </summary>
        private uint _variableDatumLength;

        /// <summary>
        /// Variable datum. This can be any number of bits long, depending on the datum.
        /// </summary>
        private uint _variableDatumBits;

        /// <summary>
        /// padding to put the record on a 64 bit boundary
        /// </summary>
        private uint _padding;

        /// <summary>
        /// Initializes a new instance of the <see cref="VariableDatum"/> class.
        /// </summary>
        public VariableDatum()
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
        public static bool operator !=(VariableDatum left, VariableDatum right)
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
        public static bool operator ==(VariableDatum left, VariableDatum right)
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

            marshalSize += 4;  // this._variableDatumID
            marshalSize += 4;  // this._variableDatumLength
            marshalSize += 4;  // this._variableDatumBits
            marshalSize += 4;  // this._padding
            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the Type of variable datum to be transmitted. 32 bit enumeration defined in EBV
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "variableDatumID")]
        public uint VariableDatumID
        {
            get
            {
                return this._variableDatumID;
            }

            set
            {
                this._variableDatumID = value;
            }
        }

        /// <summary>
        /// Gets or sets the Length, in bits, of the variable datum.
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "variableDatumLength")]
        public uint VariableDatumLength
        {
            get
            {
                return this._variableDatumLength;
            }

            set
            {
                this._variableDatumLength = value;
            }
        }

        /// <summary>
        /// Gets or sets the Variable datum. This can be any number of bits long, depending on the datum.
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "variableDatumBits")]
        public uint VariableDatumBits
        {
            get
            {
                return this._variableDatumBits;
            }

            set
            {
                this._variableDatumBits = value;
            }
        }

        /// <summary>
        /// Gets or sets the padding to put the record on a 64 bit boundary
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "padding")]
        public uint Padding
        {
            get
            {
                return this._padding;
            }

            set
            {
                this._padding = value;
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
                    dos.WriteUnsignedInt((uint)this._variableDatumID);
                    dos.WriteUnsignedInt((uint)this._variableDatumLength);
                    dos.WriteUnsignedInt((uint)this._variableDatumBits);
                    dos.WriteUnsignedInt((uint)this._padding);
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
                    this._variableDatumID = dis.ReadUnsignedInt();
                    this._variableDatumLength = dis.ReadUnsignedInt();
                    this._variableDatumBits = dis.ReadUnsignedInt();
                    this._padding = dis.ReadUnsignedInt();
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
            sb.AppendLine("<VariableDatum>");
            try
            {
                sb.AppendLine("<variableDatumID type=\"uint\">" + this._variableDatumID.ToString(CultureInfo.InvariantCulture) + "</variableDatumID>");
                sb.AppendLine("<variableDatumLength type=\"uint\">" + this._variableDatumLength.ToString(CultureInfo.InvariantCulture) + "</variableDatumLength>");
                sb.AppendLine("<variableDatumBits type=\"uint\">" + this._variableDatumBits.ToString(CultureInfo.InvariantCulture) + "</variableDatumBits>");
                sb.AppendLine("<padding type=\"uint\">" + this._padding.ToString(CultureInfo.InvariantCulture) + "</padding>");
                sb.AppendLine("</VariableDatum>");
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
            return this == obj as VariableDatum;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(VariableDatum obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            if (this._variableDatumID != obj._variableDatumID)
            {
                ivarsEqual = false;
            }

            if (this._variableDatumLength != obj._variableDatumLength)
            {
                ivarsEqual = false;
            }

            if (this._variableDatumBits != obj._variableDatumBits)
            {
                ivarsEqual = false;
            }

            if (this._padding != obj._padding)
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

            result = GenerateHash(result) ^ this._variableDatumID.GetHashCode();
            result = GenerateHash(result) ^ this._variableDatumLength.GetHashCode();
            result = GenerateHash(result) ^ this._variableDatumBits.GetHashCode();
            result = GenerateHash(result) ^ this._padding.GetHashCode();

            return result;
        }
    }
}
