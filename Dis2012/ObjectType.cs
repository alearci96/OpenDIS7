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
    /// The unique designation of an environmental object. Section 6.2.64
    /// </summary>
    [Serializable]
    [XmlRoot]
    public partial class ObjectType
    {
        /// <summary>
        /// Domain of entity (air, surface, subsurface, space, etc)
        /// </summary>
        private byte _domain;

        /// <summary>
        /// country to which the design of the entity is attributed
        /// </summary>
        private byte _objectKind;

        /// <summary>
        /// category of entity
        /// </summary>
        private byte _category;

        /// <summary>
        /// subcategory of entity
        /// </summary>
        private byte _subcategory;

        /// <summary>
        /// Initializes a new instance of the <see cref="ObjectType"/> class.
        /// </summary>
        public ObjectType()
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
        public static bool operator !=(ObjectType left, ObjectType right)
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
        public static bool operator ==(ObjectType left, ObjectType right)
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

            marshalSize += 1;  // this._domain
            marshalSize += 1;  // this._objectKind
            marshalSize += 1;  // this._category
            marshalSize += 1;  // this._subcategory
            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the Domain of entity (air, surface, subsurface, space, etc)
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "domain")]
        public byte Domain
        {
            get
            {
                return this._domain;
            }

            set
            {
                this._domain = value;
            }
        }

        /// <summary>
        /// Gets or sets the country to which the design of the entity is attributed
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "objectKind")]
        public byte ObjectKind
        {
            get
            {
                return this._objectKind;
            }

            set
            {
                this._objectKind = value;
            }
        }

        /// <summary>
        /// Gets or sets the category of entity
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "category")]
        public byte Category
        {
            get
            {
                return this._category;
            }

            set
            {
                this._category = value;
            }
        }

        /// <summary>
        /// Gets or sets the subcategory of entity
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "subcategory")]
        public byte Subcategory
        {
            get
            {
                return this._subcategory;
            }

            set
            {
                this._subcategory = value;
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
                    dos.WriteUnsignedByte((byte)this._domain);
                    dos.WriteUnsignedByte((byte)this._objectKind);
                    dos.WriteUnsignedByte((byte)this._category);
                    dos.WriteUnsignedByte((byte)this._subcategory);
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
                    this._domain = dis.ReadUnsignedByte();
                    this._objectKind = dis.ReadUnsignedByte();
                    this._category = dis.ReadUnsignedByte();
                    this._subcategory = dis.ReadUnsignedByte();
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
            sb.AppendLine("<ObjectType>");
            try
            {
                sb.AppendLine("<domain type=\"byte\">" + this._domain.ToString(CultureInfo.InvariantCulture) + "</domain>");
                sb.AppendLine("<objectKind type=\"byte\">" + this._objectKind.ToString(CultureInfo.InvariantCulture) + "</objectKind>");
                sb.AppendLine("<category type=\"byte\">" + this._category.ToString(CultureInfo.InvariantCulture) + "</category>");
                sb.AppendLine("<subcategory type=\"byte\">" + this._subcategory.ToString(CultureInfo.InvariantCulture) + "</subcategory>");
                sb.AppendLine("</ObjectType>");
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
            return this == obj as ObjectType;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(ObjectType obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            if (this._domain != obj._domain)
            {
                ivarsEqual = false;
            }

            if (this._objectKind != obj._objectKind)
            {
                ivarsEqual = false;
            }

            if (this._category != obj._category)
            {
                ivarsEqual = false;
            }

            if (this._subcategory != obj._subcategory)
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

            result = GenerateHash(result) ^ this._domain.GetHashCode();
            result = GenerateHash(result) ^ this._objectKind.GetHashCode();
            result = GenerateHash(result) ^ this._category.GetHashCode();
            result = GenerateHash(result) ^ this._subcategory.GetHashCode();

            return result;
        }
    }
}
