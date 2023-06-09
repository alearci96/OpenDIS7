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
    /// Information about elastic collisions in a DIS exercise shall be communicated using a Collision-Elastic PDU. Section 7.2.4. COMPLETE
    /// </summary>
    [Serializable]
    [XmlRoot]
    [XmlInclude(typeof(EntityID))]
    [XmlInclude(typeof(EventIdentifier))]
    [XmlInclude(typeof(Vector3Float))]
    public partial class CollisionElasticPdu : EntityInformationFamilyPdu, IEquatable<CollisionElasticPdu>
    {
        /// <summary>
        /// This field shall identify the entity that is issuing the PDU and shall be represented by an Entity Identifier record (see 6.2.28)
        /// </summary>
        private EntityID _issuingEntityID = new EntityID();

        /// <summary>
        /// This field shall identify the entity that has collided with the issuing entity. This field shall be a valid identifier of an entity or server capable of responding to the receipt of this Collision-Elastic PDU. This field shall be represented by an Entity Identifier record (see 6.2.28).
        /// </summary>
        private EntityID _collidingEntityID = new EntityID();

        /// <summary>
        /// This field shall contain an identification generated by the issuing simulation application to associate related collision events. This field shall be represented by an Event Identifier record (see 6.2.34).
        /// </summary>
        private EventIdentifier _collisionEventID = new EventIdentifier();

        /// <summary>
        /// some padding
        /// </summary>
        private short _pad;

        /// <summary>
        /// This field shall contain the velocity at the time the collision is detected at the point the collision is detected. The velocity shall be represented in world coordinates. This field shall be represented by the Linear Velocity Vector record [see 6.2.95 item c)]
        /// </summary>
        private Vector3Float _contactVelocity = new Vector3Float();

        /// <summary>
        /// This field shall contain the mass of the issuing entity and shall be represented by a 32-bit floating point number representing kilograms
        /// </summary>
        private float _mass;

        /// <summary>
        /// This field shall specify the location of the collision with respect to the entity with which the issuing entity collided. This field shall be represented by an Entity Coordinate Vector record [see 6.2.95 item a)].
        /// </summary>
        private Vector3Float _locationOfImpact = new Vector3Float();

        /// <summary>
        /// These six records represent the six independent components of a positive semi-definite matrix formed by pre-multiplying and post-multiplying the tensor of inertia, by the anti-symmetric matrix generated by the moment arm, and shall be represented by 32-bit floating point numbers (see 5.3.4.4)
        /// </summary>
        private float _collisionIntermediateResultXX;

        /// <summary>
        /// tensor values
        /// </summary>
        private float _collisionIntermediateResultXY;

        /// <summary>
        /// tensor values
        /// </summary>
        private float _collisionIntermediateResultXZ;

        /// <summary>
        /// tensor values
        /// </summary>
        private float _collisionIntermediateResultYY;

        /// <summary>
        /// tensor values
        /// </summary>
        private float _collisionIntermediateResultYZ;

        /// <summary>
        /// tensor values
        /// </summary>
        private float _collisionIntermediateResultZZ;

        /// <summary>
        /// This record shall represent the normal vector to the surface at the point of collision detection. The surface normal shall be represented in world coordinates. This field shall be represented by an Entity Coordinate Vector record [see 6.2.95 item a)].
        /// </summary>
        private Vector3Float _unitSurfaceNormal = new Vector3Float();

        /// <summary>
        /// This field shall represent the degree to which energy is conserved in a collision and shall be represented by a 32-bit floating point number. In addition, it represents a free parameter by which simulation application developers may “tune” their collision interactions.
        /// </summary>
        private float _coefficientOfRestitution;

        /// <summary>
        /// Initializes a new instance of the <see cref="CollisionElasticPdu"/> class.
        /// </summary>
        public CollisionElasticPdu()
        {
            PduType = (byte)66;
            ProtocolFamily = (byte)1;
        }

        /// <summary>
        /// Implements the operator !=.
        /// </summary>
        /// <param name="left">The left operand.</param>
        /// <param name="right">The right operand.</param>
        /// <returns>
        /// 	<c>true</c> if operands are not equal; otherwise, <c>false</c>.
        /// </returns>
        public static bool operator !=(CollisionElasticPdu left, CollisionElasticPdu right)
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
        public static bool operator ==(CollisionElasticPdu left, CollisionElasticPdu right)
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
            marshalSize += this._issuingEntityID.GetMarshalledSize();  // this._issuingEntityID
            marshalSize += this._collidingEntityID.GetMarshalledSize();  // this._collidingEntityID
            marshalSize += this._collisionEventID.GetMarshalledSize();  // this._collisionEventID
            marshalSize += 2;  // this._pad
            marshalSize += this._contactVelocity.GetMarshalledSize();  // this._contactVelocity
            marshalSize += 4;  // this._mass
            marshalSize += this._locationOfImpact.GetMarshalledSize();  // this._locationOfImpact
            marshalSize += 4;  // this._collisionIntermediateResultXX
            marshalSize += 4;  // this._collisionIntermediateResultXY
            marshalSize += 4;  // this._collisionIntermediateResultXZ
            marshalSize += 4;  // this._collisionIntermediateResultYY
            marshalSize += 4;  // this._collisionIntermediateResultYZ
            marshalSize += 4;  // this._collisionIntermediateResultZZ
            marshalSize += this._unitSurfaceNormal.GetMarshalledSize();  // this._unitSurfaceNormal
            marshalSize += 4;  // this._coefficientOfRestitution
            return marshalSize;
        }

        /// <summary>
        /// Gets or sets the This field shall identify the entity that is issuing the PDU and shall be represented by an Entity Identifier record (see 6.2.28)
        /// </summary>
        [XmlElement(Type = typeof(EntityID), ElementName = "issuingEntityID")]
        public EntityID IssuingEntityID
        {
            get
            {
                return this._issuingEntityID;
            }

            set
            {
                this._issuingEntityID = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall identify the entity that has collided with the issuing entity. This field shall be a valid identifier of an entity or server capable of responding to the receipt of this Collision-Elastic PDU. This field shall be represented by an Entity Identifier record (see 6.2.28).
        /// </summary>
        [XmlElement(Type = typeof(EntityID), ElementName = "collidingEntityID")]
        public EntityID CollidingEntityID
        {
            get
            {
                return this._collidingEntityID;
            }

            set
            {
                this._collidingEntityID = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall contain an identification generated by the issuing simulation application to associate related collision events. This field shall be represented by an Event Identifier record (see 6.2.34).
        /// </summary>
        [XmlElement(Type = typeof(EventIdentifier), ElementName = "collisionEventID")]
        public EventIdentifier CollisionEventID
        {
            get
            {
                return this._collisionEventID;
            }

            set
            {
                this._collisionEventID = value;
            }
        }

        /// <summary>
        /// Gets or sets the some padding
        /// </summary>
        [XmlElement(Type = typeof(short), ElementName = "pad")]
        public short Pad
        {
            get
            {
                return this._pad;
            }

            set
            {
                this._pad = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall contain the velocity at the time the collision is detected at the point the collision is detected. The velocity shall be represented in world coordinates. This field shall be represented by the Linear Velocity Vector record [see 6.2.95 item c)]
        /// </summary>
        [XmlElement(Type = typeof(Vector3Float), ElementName = "contactVelocity")]
        public Vector3Float ContactVelocity
        {
            get
            {
                return this._contactVelocity;
            }

            set
            {
                this._contactVelocity = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall contain the mass of the issuing entity and shall be represented by a 32-bit floating point number representing kilograms
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "mass")]
        public float Mass
        {
            get
            {
                return this._mass;
            }

            set
            {
                this._mass = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall specify the location of the collision with respect to the entity with which the issuing entity collided. This field shall be represented by an Entity Coordinate Vector record [see 6.2.95 item a)].
        /// </summary>
        [XmlElement(Type = typeof(Vector3Float), ElementName = "locationOfImpact")]
        public Vector3Float LocationOfImpact
        {
            get
            {
                return this._locationOfImpact;
            }

            set
            {
                this._locationOfImpact = value;
            }
        }

        /// <summary>
        /// Gets or sets the These six records represent the six independent components of a positive semi-definite matrix formed by pre-multiplying and post-multiplying the tensor of inertia, by the anti-symmetric matrix generated by the moment arm, and shall be represented by 32-bit floating point numbers (see 5.3.4.4)
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultXX")]
        public float CollisionIntermediateResultXX
        {
            get
            {
                return this._collisionIntermediateResultXX;
            }

            set
            {
                this._collisionIntermediateResultXX = value;
            }
        }

        /// <summary>
        /// Gets or sets the tensor values
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultXY")]
        public float CollisionIntermediateResultXY
        {
            get
            {
                return this._collisionIntermediateResultXY;
            }

            set
            {
                this._collisionIntermediateResultXY = value;
            }
        }

        /// <summary>
        /// Gets or sets the tensor values
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultXZ")]
        public float CollisionIntermediateResultXZ
        {
            get
            {
                return this._collisionIntermediateResultXZ;
            }

            set
            {
                this._collisionIntermediateResultXZ = value;
            }
        }

        /// <summary>
        /// Gets or sets the tensor values
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultYY")]
        public float CollisionIntermediateResultYY
        {
            get
            {
                return this._collisionIntermediateResultYY;
            }

            set
            {
                this._collisionIntermediateResultYY = value;
            }
        }

        /// <summary>
        /// Gets or sets the tensor values
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultYZ")]
        public float CollisionIntermediateResultYZ
        {
            get
            {
                return this._collisionIntermediateResultYZ;
            }

            set
            {
                this._collisionIntermediateResultYZ = value;
            }
        }

        /// <summary>
        /// Gets or sets the tensor values
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "collisionIntermediateResultZZ")]
        public float CollisionIntermediateResultZZ
        {
            get
            {
                return this._collisionIntermediateResultZZ;
            }

            set
            {
                this._collisionIntermediateResultZZ = value;
            }
        }

        /// <summary>
        /// Gets or sets the This record shall represent the normal vector to the surface at the point of collision detection. The surface normal shall be represented in world coordinates. This field shall be represented by an Entity Coordinate Vector record [see 6.2.95 item a)].
        /// </summary>
        [XmlElement(Type = typeof(Vector3Float), ElementName = "unitSurfaceNormal")]
        public Vector3Float UnitSurfaceNormal
        {
            get
            {
                return this._unitSurfaceNormal;
            }

            set
            {
                this._unitSurfaceNormal = value;
            }
        }

        /// <summary>
        /// Gets or sets the This field shall represent the degree to which energy is conserved in a collision and shall be represented by a 32-bit floating point number. In addition, it represents a free parameter by which simulation application developers may “tune” their collision interactions.
        /// </summary>
        [XmlElement(Type = typeof(float), ElementName = "coefficientOfRestitution")]
        public float CoefficientOfRestitution
        {
            get
            {
                return this._coefficientOfRestitution;
            }

            set
            {
                this._coefficientOfRestitution = value;
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
                    this._issuingEntityID.Marshal(dos);
                    this._collidingEntityID.Marshal(dos);
                    this._collisionEventID.Marshal(dos);
                    dos.WriteShort((short)this._pad);
                    this._contactVelocity.Marshal(dos);
                    dos.WriteFloat((float)this._mass);
                    this._locationOfImpact.Marshal(dos);
                    dos.WriteFloat((float)this._collisionIntermediateResultXX);
                    dos.WriteFloat((float)this._collisionIntermediateResultXY);
                    dos.WriteFloat((float)this._collisionIntermediateResultXZ);
                    dos.WriteFloat((float)this._collisionIntermediateResultYY);
                    dos.WriteFloat((float)this._collisionIntermediateResultYZ);
                    dos.WriteFloat((float)this._collisionIntermediateResultZZ);
                    this._unitSurfaceNormal.Marshal(dos);
                    dos.WriteFloat((float)this._coefficientOfRestitution);
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
                    this._issuingEntityID.Unmarshal(dis);
                    this._collidingEntityID.Unmarshal(dis);
                    this._collisionEventID.Unmarshal(dis);
                    this._pad = dis.ReadShort();
                    this._contactVelocity.Unmarshal(dis);
                    this._mass = dis.ReadFloat();
                    this._locationOfImpact.Unmarshal(dis);
                    this._collisionIntermediateResultXX = dis.ReadFloat();
                    this._collisionIntermediateResultXY = dis.ReadFloat();
                    this._collisionIntermediateResultXZ = dis.ReadFloat();
                    this._collisionIntermediateResultYY = dis.ReadFloat();
                    this._collisionIntermediateResultYZ = dis.ReadFloat();
                    this._collisionIntermediateResultZZ = dis.ReadFloat();
                    this._unitSurfaceNormal.Unmarshal(dis);
                    this._coefficientOfRestitution = dis.ReadFloat();
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
            sb.AppendLine("<CollisionElasticPdu>");
            base.Reflection(sb);
            try
            {
                sb.AppendLine("<issuingEntityID>");
                this._issuingEntityID.Reflection(sb);
                sb.AppendLine("</issuingEntityID>");
                sb.AppendLine("<collidingEntityID>");
                this._collidingEntityID.Reflection(sb);
                sb.AppendLine("</collidingEntityID>");
                sb.AppendLine("<collisionEventID>");
                this._collisionEventID.Reflection(sb);
                sb.AppendLine("</collisionEventID>");
                sb.AppendLine("<pad type=\"short\">" + this._pad.ToString(CultureInfo.InvariantCulture) + "</pad>");
                sb.AppendLine("<contactVelocity>");
                this._contactVelocity.Reflection(sb);
                sb.AppendLine("</contactVelocity>");
                sb.AppendLine("<mass type=\"float\">" + this._mass.ToString(CultureInfo.InvariantCulture) + "</mass>");
                sb.AppendLine("<locationOfImpact>");
                this._locationOfImpact.Reflection(sb);
                sb.AppendLine("</locationOfImpact>");
                sb.AppendLine("<collisionIntermediateResultXX type=\"float\">" + this._collisionIntermediateResultXX.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultXX>");
                sb.AppendLine("<collisionIntermediateResultXY type=\"float\">" + this._collisionIntermediateResultXY.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultXY>");
                sb.AppendLine("<collisionIntermediateResultXZ type=\"float\">" + this._collisionIntermediateResultXZ.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultXZ>");
                sb.AppendLine("<collisionIntermediateResultYY type=\"float\">" + this._collisionIntermediateResultYY.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultYY>");
                sb.AppendLine("<collisionIntermediateResultYZ type=\"float\">" + this._collisionIntermediateResultYZ.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultYZ>");
                sb.AppendLine("<collisionIntermediateResultZZ type=\"float\">" + this._collisionIntermediateResultZZ.ToString(CultureInfo.InvariantCulture) + "</collisionIntermediateResultZZ>");
                sb.AppendLine("<unitSurfaceNormal>");
                this._unitSurfaceNormal.Reflection(sb);
                sb.AppendLine("</unitSurfaceNormal>");
                sb.AppendLine("<coefficientOfRestitution type=\"float\">" + this._coefficientOfRestitution.ToString(CultureInfo.InvariantCulture) + "</coefficientOfRestitution>");
                sb.AppendLine("</CollisionElasticPdu>");
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
            return this == obj as CollisionElasticPdu;
        }

        /// <summary>
        /// Compares for reference AND value equality.
        /// </summary>
        /// <param name="obj">The object to compare with this instance.</param>
        /// <returns>
        /// 	<c>true</c> if both operands are equal; otherwise, <c>false</c>.
        /// </returns>
        public bool Equals(CollisionElasticPdu obj)
        {
            bool ivarsEqual = true;

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            ivarsEqual = base.Equals(obj);

            if (!this._issuingEntityID.Equals(obj._issuingEntityID))
            {
                ivarsEqual = false;
            }

            if (!this._collidingEntityID.Equals(obj._collidingEntityID))
            {
                ivarsEqual = false;
            }

            if (!this._collisionEventID.Equals(obj._collisionEventID))
            {
                ivarsEqual = false;
            }

            if (this._pad != obj._pad)
            {
                ivarsEqual = false;
            }

            if (!this._contactVelocity.Equals(obj._contactVelocity))
            {
                ivarsEqual = false;
            }

            if (this._mass != obj._mass)
            {
                ivarsEqual = false;
            }

            if (!this._locationOfImpact.Equals(obj._locationOfImpact))
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultXX != obj._collisionIntermediateResultXX)
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultXY != obj._collisionIntermediateResultXY)
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultXZ != obj._collisionIntermediateResultXZ)
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultYY != obj._collisionIntermediateResultYY)
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultYZ != obj._collisionIntermediateResultYZ)
            {
                ivarsEqual = false;
            }

            if (this._collisionIntermediateResultZZ != obj._collisionIntermediateResultZZ)
            {
                ivarsEqual = false;
            }

            if (!this._unitSurfaceNormal.Equals(obj._unitSurfaceNormal))
            {
                ivarsEqual = false;
            }

            if (this._coefficientOfRestitution != obj._coefficientOfRestitution)
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

            result = GenerateHash(result) ^ base.GetHashCode();

            result = GenerateHash(result) ^ this._issuingEntityID.GetHashCode();
            result = GenerateHash(result) ^ this._collidingEntityID.GetHashCode();
            result = GenerateHash(result) ^ this._collisionEventID.GetHashCode();
            result = GenerateHash(result) ^ this._pad.GetHashCode();
            result = GenerateHash(result) ^ this._contactVelocity.GetHashCode();
            result = GenerateHash(result) ^ this._mass.GetHashCode();
            result = GenerateHash(result) ^ this._locationOfImpact.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultXX.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultXY.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultXZ.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultYY.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultYZ.GetHashCode();
            result = GenerateHash(result) ^ this._collisionIntermediateResultZZ.GetHashCode();
            result = GenerateHash(result) ^ this._unitSurfaceNormal.GetHashCode();
            result = GenerateHash(result) ^ this._coefficientOfRestitution.GetHashCode();

            return result;
        }
    }
}
