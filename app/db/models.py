from sqlalchemy import Column, String, DateTime, DECIMAL, Boolean, ForeignKey, Integer, Text
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.database import Base


class BaseEntity(Base):
    __abstract__ = True
    Id = Column(UNIQUEIDENTIFIER, primary_key=True, default=uuid.uuid4)


class BaseAuditableEntity(BaseEntity):
    __abstract__ = True
    
    CreatedOn = Column(DateTime, nullable=False, default=datetime.utcnow)
    CreatedBy = Column(UNIQUEIDENTIFIER, nullable=True)
    LastModifiedOn = Column(DateTime, nullable=True)
    LastModifiedBy = Column(UNIQUEIDENTIFIER, nullable=True)


class TransactionCategory(BaseAuditableEntity):
    __tablename__ = "TransactionCategories"
    
    Name = Column(String(255), nullable=False)
    Description = Column(Text, nullable=True)
    Type = Column(Integer, nullable=False)
    
    Transactions = relationship("Transaction", back_populates="Category")
    BudgetDetails = relationship("BudgetDetail", back_populates="Category")


class Transaction(BaseAuditableEntity):
    __tablename__ = "Transactions"
    
    Type = Column(Integer, nullable=False)
    CategoryId = Column(UNIQUEIDENTIFIER, ForeignKey("TransactionCategories.Id"), nullable=False)
    Amount = Column(DECIMAL(18, 2), nullable=False)
    Date = Column(DateTime, nullable=False)
    Notes = Column(Text, nullable=True)
    
    Category = relationship("TransactionCategory", back_populates="Transactions")


class SavingGoal(BaseAuditableEntity):
    __tablename__ = "SavingGoals"
    
    Name = Column(String(255), nullable=False)
    TargetAmount = Column(DECIMAL(18, 2), nullable=False)
    CurrentAmount = Column(DECIMAL(18, 2), nullable=False, default=0)
    Deadline = Column(DateTime, nullable=True)
    IsCompleted = Column(Boolean, nullable=False, default=False)
    Notes = Column(Text, nullable=True)
    
    Contributions = relationship("Contribution", back_populates="SavingGoal")


class Contribution(BaseAuditableEntity):
    __tablename__ = "Contributions"
    
    SavingGoalId = Column(UNIQUEIDENTIFIER, ForeignKey("SavingGoals.Id"), nullable=False)
    Amount = Column(DECIMAL(18, 2), nullable=False)
    Date = Column(DateTime, nullable=False)
    Notes = Column(Text, nullable=True)
    
    SavingGoal = relationship("SavingGoal", back_populates="Contributions")


class Budget(BaseAuditableEntity):
    __tablename__ = "Budgets"
    
    Name = Column(String(255), nullable=False)
    TotalLimit = Column(DECIMAL(18, 2), nullable=False)
    TotalSpent = Column(DECIMAL(18, 2), nullable=False, default=0)
    StartDate = Column(DateTime, nullable=False)
    EndDate = Column(DateTime, nullable=False)
    
    BudgetDetails = relationship("BudgetDetail", back_populates="Budget")


class BudgetDetail(BaseAuditableEntity):
    __tablename__ = "BudgetDetails"
    
    BudgetId = Column(UNIQUEIDENTIFIER, ForeignKey("Budgets.Id"), nullable=False)
    CategoryId = Column(UNIQUEIDENTIFIER, ForeignKey("TransactionCategories.Id"), nullable=False)
    Limit = Column(DECIMAL(18, 2), nullable=False)
    Spent = Column(DECIMAL(18, 2), nullable=False, default=0)
    
    Budget = relationship("Budget", back_populates="BudgetDetails")
    Category = relationship("TransactionCategory", back_populates="BudgetDetails")


class Notification(BaseAuditableEntity):
    __tablename__ = "Notifications"
    
    Type = Column(Integer, nullable=False)
    Title = Column(String(255), nullable=False)
    Message = Column(Text, nullable=False)
    IsRead = Column(Boolean, nullable=False, default=False)
    ReadAt = Column(DateTime, nullable=True)

