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
    /// First part of a simulation management (SIMAN) PDU and SIMAN-Reliability (SIMAN-R) PDU. Sectionn 6.2.81
    /// </summary>
    [Serializable]
    [XmlRoot]
    [XmlInclude(typeof(PduHeader))]
    [XmlInclude(typeof(EntityID))]
    public partial class SimulationManagementPduHeader
    {
        /// <summary>
        /// Conventional PDU header
        /// </summary>
        private PduHeader _pduHeader = new PduHeader();

        /// <summary>
        /// IDs the simulation or entity, etiehr a simulation or an entity. Either 6.2.80 or 6.2.28
        /// </summary>
        private EntityID _originatingID = new EntityID();

        /// <summary>
        /// simulation, all simulations, a special ID, or an entity. See 5.6.5 and 5.12.4
        /// </summary>
        private EntityID _recevingID = new EntityID();

        /// <summary>
        /// Initializes a new instance of the <see cref="SimulationManagementPduHeader"/> class.
        /// </summary>
        public SimulationManagementPduHeader()
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
        public static bool operator !=(SimulationManagementPduHeader left, SimulationManagementPduHeader right)
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
        public static bool operator ==(SimulationManagementPduHeader left, SimulationManagementPduHeader right)
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

            marshalSize += this._pduHeader.GetMarshalledSize();  // this._pduHeader
            marshalSize += this._originatingID.GetMarshalledSize();  // this._originatingID
            marshalSize += this._recevingID.GetMarshalledSize();  // this._recevingID
            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the Conventional PDU header
        /// </summary>
        [XmlElement(Type = typeof(PduHeader), ElementName = "pduHeader")]
        public PduHeader PduHeader
        {
            get
            {
                return this._pduHeader;
            }

            set
            {
                this._pduHeader = value;
            }
        }

        /// <summary>
        /// Gets or sets the IDs the simulation or entity, etiehr a simulation or an entity. Either 6.2.80 or 6.2.28
        /// </summary>
        [XmlElement(Type = typeof(EntityID), ElementName = "originatingID")]
        public EntityID OriginatingID
        {
            get
            {
                return this._originatingID;
            }

            set
            {
                this._originatingID = value;
            }
        }

        /// <summary>
        /// Gets or sets the simulation, all simulations, a special ID, or an entity. See 5.6.5 and 5.12.4
        /// </summary>
        [XmlElement(Type = typeof(EntityID), ElementName = "recevingID")]
        public EntityID RecevingID
        {
            get
            {
                return this._recevingID;
            }

            set
            {
                this._recevingID = value;
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
                    this._pduHeader.Marshal(dos);
                    this._originatingID.Marshal(dos);
                    this._recevingID.Marshal(dos);
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
                    this._pduHeader.Unmarshal(dis);
                    this._originatingID.Unmarshal(dis);
                    this._recevingID.Unmarshal(dis);
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
            sb.AppendLine("<SimulationManagementPduHeader>");
            try
            {
                sb.AppendLine("<pduHeader>");
                this._pduHeader.Reflection(sb);
                sb.AppendLine("</pduHeader>");
                sb.AppendLine("<originatingID>");
                this._originatingID.Reflection(sb);
                sb.AppendLine("</originatingID>");
                sb.AppendLine("<recevingID>");
                this._recevingID.Reflection(sb);
                sb.AppendLine("</recevingID>");
                sb.AppendLine("</SimulationManagementPduHeader>");
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
            return this == obj as SimulationManagementPduHeader;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(SimulationManagementPduHeader obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            if (!this._pduHeader.Equals(obj._pduHeader))
            {
                ivarsEqual = false;
            }

            if (!this._originatingID.Equals(obj._originatingID))
            {
                ivarsEqual = false;
            }

            if (!this._recevingID.Equals(obj._recevingID))
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

            result = GenerateHash(result) ^ this._pduHeader.GetHashCode();
            result = GenerateHash(result) ^ this._originatingID.GetHashCode();
            result = GenerateHash(result) ^ this._recevingID.GetHashCode();

            return result;
        }
    }
}
