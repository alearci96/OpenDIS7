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
    /// Section 5.3.12.10: issued in response to a data query R or set dataR pdu. Needs manual intervention      to fix padding on variable datums. UNFINSIHED
    /// </summary>
    [Serializable]
    [XmlRoot]
    [XmlInclude(typeof(FixedDatum))]
    [XmlInclude(typeof(VariableDatum))]
    public partial class DataReliablePdu : SimulationManagementWithReliabilityFamilyPdu, IEquatable<DataReliablePdu>
    {
        /// <summary>
        /// Request ID
        /// </summary>
        private uint _requestID;

        /// <summary>
        /// level of reliability service used for this transaction
        /// </summary>
        private byte _requiredReliabilityService;

        /// <summary>
        /// padding
        /// </summary>
        private ushort _pad1;

        /// <summary>
        /// padding
        /// </summary>
        private byte _pad2;

        /// <summary>
        /// Fixed datum record count
        /// </summary>
        private uint _numberOfFixedDatumRecords;

        /// <summary>
        /// variable datum record count
        /// </summary>
        private uint _numberOfVariableDatumRecords;

        /// <summary>
        /// Fixed datum records
        /// </summary>
        private List<FixedDatum> _fixedDatumRecords = new List<FixedDatum>();

        /// <summary>
        /// Variable datum records
        /// </summary>
        private List<VariableDatum> _variableDatumRecords = new List<VariableDatum>();

        /// <summary>
        /// Initializes a new instance of the <see cref="DataReliablePdu"/> class.
        /// </summary>
        public DataReliablePdu()
        {
            PduType = (byte)60;
        }

        /// <summary>
        /// Implements the operator !=.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        /// 	<c>true</c> if operands are not equal; otherwise, <c>false</c>.
        /// </returns>
        public static bool operator !=(DataReliablePdu left, DataReliablePdu right)
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
        public static bool operator ==(DataReliablePdu left, DataReliablePdu right)
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

        public override int GetMarshalledSize()
        {
            int marshalSize = 0; 

            marshalSize = base.GetMarshalledSize();
            marshalSize += 4;  // this._requestID
            marshalSize += 1;  // this._requiredReliabilityService
            marshalSize += 2;  // this._pad1
            marshalSize += 1;  // this._pad2
            marshalSize += 4;  // this._numberOfFixedDatumRecords
            marshalSize += 4;  // this._numberOfVariableDatumRecords
            for (int idx = 0; idx < this._fixedDatumRecords.Count; idx++)
            {
                FixedDatum listElement = (FixedDatum)this._fixedDatumRecords[idx];
                marshalSize += listElement.GetMarshalledSize();
            }

            for (int idx = 0; idx < this._variableDatumRecords.Count; idx++)
            {
                VariableDatum listElement = (VariableDatum)this._variableDatumRecords[idx];
                marshalSize += listElement.GetMarshalledSize();
            }

            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the Request ID
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "requestID")]
        public uint RequestID
        {
            get
            {
                return this._requestID;
            }

            set
            {
                this._requestID = value;
            }
        }

        /// <summary>
        /// Gets or sets the level of reliability service used for this transaction
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "requiredReliabilityService")]
        public byte RequiredReliabilityService
        {
            get
            {
                return this._requiredReliabilityService;
            }

            set
            {
                this._requiredReliabilityService = value;
            }
        }

        /// <summary>
        /// Gets or sets the padding
        /// </summary>
        [XmlElement(Type = typeof(ushort), ElementName = "pad1")]
        public ushort Pad1
        {
            get
            {
                return this._pad1;
            }

            set
            {
                this._pad1 = value;
            }
        }

        /// <summary>
        /// Gets or sets the padding
        /// </summary>
        [XmlElement(Type = typeof(byte), ElementName = "pad2")]
        public byte Pad2
        {
            get
            {
                return this._pad2;
            }

            set
            {
                this._pad2 = value;
            }
        }

        /// <summary>
        /// Note that setting this value will not change the marshalled value. The list whose length this describes is used for that purpose.
        /// The getnumberOfFixedDatumRecords method will also be based on the actual list length rather than this value. 
        /// The method is simply here for completeness and should not be used for any computations.
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "numberOfFixedDatumRecords")]
        public uint NumberOfFixedDatumRecords
        {
            get
            {
                return this._numberOfFixedDatumRecords;
            }

            set
            {
                this._numberOfFixedDatumRecords = value;
            }
        }

        /// <summary>
        /// Note that setting this value will not change the marshalled value. The list whose length this describes is used for that purpose.
        /// The getnumberOfVariableDatumRecords method will also be based on the actual list length rather than this value. 
        /// The method is simply here for completeness and should not be used for any computations.
        /// </summary>
        [XmlElement(Type = typeof(uint), ElementName = "numberOfVariableDatumRecords")]
        public uint NumberOfVariableDatumRecords
        {
            get
            {
                return this._numberOfVariableDatumRecords;
            }

            set
            {
                this._numberOfVariableDatumRecords = value;
            }
        }

        /// <summary>
        /// Gets or sets the Fixed datum records
        /// </summary>
        [XmlElement(ElementName = "fixedDatumRecordsList", Type = typeof(List<FixedDatum>))]
        public List<FixedDatum> FixedDatumRecords
        {
            get
            {
                return this._fixedDatumRecords;
            }
        }

        /// <summary>
        /// Gets or sets the Variable datum records
        /// </summary>
        [XmlElement(ElementName = "variableDatumRecordsList", Type = typeof(List<VariableDatum>))]
        public List<VariableDatum> VariableDatumRecords
        {
            get
            {
                return this._variableDatumRecords;
            }
        }

        /// <summary>
        /// Automatically sets the length of the marshalled data, then calls the marshal method.
        /// </summary>
        /// <param name="dos">The DataOutputStream instance to which the PDU is marshaled.</param>
        public override void MarshalAutoLengthSet(DataOutputStream dos)
        {
            // Set the length prior to marshalling data
            this.Length = (ushort)this.GetMarshalledSize();
            this.Marshal(dos);
        }

        /// <summary>
        /// Marshal the data to the DataOutputStream.  Note: Length needs to be set before calling this method
        /// </summary>
        /// <param name="dos">The DataOutputStream instance to which the PDU is marshaled.</param>
        [SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes", Justification = "Due to ignoring errors.")]
        public override void Marshal(DataOutputStream dos)
        {
            base.Marshal(dos);
            if (dos != null)
            {
                try
                {
                    dos.WriteUnsignedInt((uint)this._requestID);
                    dos.WriteUnsignedByte((byte)this._requiredReliabilityService);
                    dos.WriteUnsignedShort((ushort)this._pad1);
                    dos.WriteUnsignedByte((byte)this._pad2);
                    dos.WriteUnsignedInt((uint)this._fixedDatumRecords.Count);
                    dos.WriteUnsignedInt((uint)this._variableDatumRecords.Count);

                    for (int idx = 0; idx < this._fixedDatumRecords.Count; idx++)
                    {
                        FixedDatum aFixedDatum = (FixedDatum)this._fixedDatumRecords[idx];
                        aFixedDatum.Marshal(dos);
                    }

                    for (int idx = 0; idx < this._variableDatumRecords.Count; idx++)
                    {
                        VariableDatum aVariableDatum = (VariableDatum)this._variableDatumRecords[idx];
                        aVariableDatum.Marshal(dos);
                    }
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
        public override void Unmarshal(DataInputStream dis)
        {
            base.Unmarshal(dis);

            if (dis != null)
            {
                try
                {
                    this._requestID = dis.ReadUnsignedInt();
                    this._requiredReliabilityService = dis.ReadUnsignedByte();
                    this._pad1 = dis.ReadUnsignedShort();
                    this._pad2 = dis.ReadUnsignedByte();
                    this._numberOfFixedDatumRecords = dis.ReadUnsignedInt();
                    this._numberOfVariableDatumRecords = dis.ReadUnsignedInt();
                    for (int idx = 0; idx < this.NumberOfFixedDatumRecords; idx++)
                    {
                        FixedDatum anX = new FixedDatum();
                        anX.Unmarshal(dis);
                        this._fixedDatumRecords.Add(anX);
                    };

                    for (int idx = 0; idx < this.NumberOfVariableDatumRecords; idx++)
                    {
                        VariableDatum anX = new VariableDatum();
                        anX.Unmarshal(dis);
                        this._variableDatumRecords.Add(anX);
                    };

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
        public override void Reflection(StringBuilder sb)
        {
            sb.AppendLine("<DataReliablePdu>");
            base.Reflection(sb);
            try
            {
                sb.AppendLine("<requestID type=\"uint\">" + this._requestID.ToString(CultureInfo.InvariantCulture) + "</requestID>");
                sb.AppendLine("<requiredReliabilityService type=\"byte\">" + this._requiredReliabilityService.ToString(CultureInfo.InvariantCulture) + "</requiredReliabilityService>");
                sb.AppendLine("<pad1 type=\"ushort\">" + this._pad1.ToString(CultureInfo.InvariantCulture) + "</pad1>");
                sb.AppendLine("<pad2 type=\"byte\">" + this._pad2.ToString(CultureInfo.InvariantCulture) + "</pad2>");
                sb.AppendLine("<fixedDatumRecords type=\"uint\">" + this._fixedDatumRecords.Count.ToString(CultureInfo.InvariantCulture) + "</fixedDatumRecords>");
                sb.AppendLine("<variableDatumRecords type=\"uint\">" + this._variableDatumRecords.Count.ToString(CultureInfo.InvariantCulture) + "</variableDatumRecords>");
                for (int idx = 0; idx < this._fixedDatumRecords.Count; idx++)
                {
                    sb.AppendLine("<fixedDatumRecords" + idx.ToString(CultureInfo.InvariantCulture) + " type=\"FixedDatum\">");
                    FixedDatum aFixedDatum = (FixedDatum)this._fixedDatumRecords[idx];
                    aFixedDatum.Reflection(sb);
                    sb.AppendLine("</fixedDatumRecords" + idx.ToString(CultureInfo.InvariantCulture) + ">");
                }

                for (int idx = 0; idx < this._variableDatumRecords.Count; idx++)
                {
                    sb.AppendLine("<variableDatumRecords" + idx.ToString(CultureInfo.InvariantCulture) + " type=\"VariableDatum\">");
                    VariableDatum aVariableDatum = (VariableDatum)this._variableDatumRecords[idx];
                    aVariableDatum.Reflection(sb);
                    sb.AppendLine("</variableDatumRecords" + idx.ToString(CultureInfo.InvariantCulture) + ">");
                }

                sb.AppendLine("</DataReliablePdu>");
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
            return this == obj as DataReliablePdu;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(DataReliablePdu obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            ivarsEqual = base.Equals(obj);

            if (this._requestID != obj._requestID)
            {
                ivarsEqual = false;
            }

            if (this._requiredReliabilityService != obj._requiredReliabilityService)
            {
                ivarsEqual = false;
            }

            if (this._pad1 != obj._pad1)
            {
                ivarsEqual = false;
            }

            if (this._pad2 != obj._pad2)
            {
                ivarsEqual = false;
            }

            if (this._numberOfFixedDatumRecords != obj._numberOfFixedDatumRecords)
            {
                ivarsEqual = false;
            }

            if (this._numberOfVariableDatumRecords != obj._numberOfVariableDatumRecords)
            {
                ivarsEqual = false;
            }

            if (this._fixedDatumRecords.Count != obj._fixedDatumRecords.Count)
            {
                ivarsEqual = false;
            }

            if (ivarsEqual)
            {
                for (int idx = 0; idx < this._fixedDatumRecords.Count; idx++)
                {
                    if (!this._fixedDatumRecords[idx].Equals(obj._fixedDatumRecords[idx]))
                    {
                        ivarsEqual = false;
                    }
                }
            }

            if (this._variableDatumRecords.Count != obj._variableDatumRecords.Count)
            {
                ivarsEqual = false;
            }

            if (ivarsEqual)
            {
                for (int idx = 0; idx < this._variableDatumRecords.Count; idx++)
                {
                    if (!this._variableDatumRecords[idx].Equals(obj._variableDatumRecords[idx]))
                    {
                        ivarsEqual = false;
                    }
                }
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

            result = GenerateHash(result) ^ base.GetHashCode();

            result = GenerateHash(result) ^ this._requestID.GetHashCode();
            result = GenerateHash(result) ^ this._requiredReliabilityService.GetHashCode();
            result = GenerateHash(result) ^ this._pad1.GetHashCode();
            result = GenerateHash(result) ^ this._pad2.GetHashCode();
            result = GenerateHash(result) ^ this._numberOfFixedDatumRecords.GetHashCode();
            result = GenerateHash(result) ^ this._numberOfVariableDatumRecords.GetHashCode();

            if (this._fixedDatumRecords.Count > 0)
            {
                for (int idx = 0; idx < this._fixedDatumRecords.Count; idx++)
                {
                    result = GenerateHash(result) ^ this._fixedDatumRecords[idx].GetHashCode();
                }
            }

            if (this._variableDatumRecords.Count > 0)
            {
                for (int idx = 0; idx < this._variableDatumRecords.Count; idx++)
                {
                    result = GenerateHash(result) ^ this._variableDatumRecords[idx].GetHashCode();
                }
            }

            return result;
        }
    }
}
